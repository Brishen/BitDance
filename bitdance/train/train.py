import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

from bitdance.train.fsdp_utils import FSDPConfig, fsdp_wrapper, FSDPCheckpoint
from bitdance.modeling.mllm import MLLModel
from bitdance.train.dataset import BitDanceDataset, BitDanceCollator
from transformers import AutoTokenizer

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        # Single GPU / CPU
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # Logging (only on rank 0)
    if rank == 0:
        if config.training.wandb_project:
            wandb.init(
                project=config.training.wandb_project,
                name=config.training.wandb_name,
                config=OmegaConf.to_container(config, resolve=True),
                mode="offline" if config.training.wandb_offline else "online",
            )
        print(f"Config loaded: {config}")

    # Build Model
    # Since MLLModel loads checkpoints (LLM, etc.), we might need to handle this carefully in distributed setting.
    # Usually we load on CPU first or on rank 0 and broadcast, but FSDP handles sharding.
    # The MLLModel __init__ loads pretrained weights.
    # To avoid OOM on CPU if loading full model multiple times, or to avoid race conditions downloading:
    # We can rely on 'download' utility in modeling.mllm imports utils.fs.download which likely handles caching.

    # We create model on CPU (meta device) or CPU then wrap.
    # MLLModel loads weights in __init__.
    # For FSDP, it's often better to init on meta device if model is huge, but here we might just init on CPU.

    model = MLLModel(config.model)

    # Enable gradient checkpointing if needed
    if hasattr(model.llm_model.model, "gradient_checkpointing_enable"):
        model.llm_model.model.gradient_checkpointing_enable()

    # FSDP Wrap
    if torch.cuda.is_available():
        fsdp_cfg = FSDPConfig(
            sharding_strategy=config.training.sharding_strategy,
            backward_prefetch=config.training.backward_prefetch,
            cpu_offload=config.training.cpu_offload,
            num_replicate=config.training.num_replicate,
            num_shard=config.training.num_shard,
        )

        model = fsdp_wrapper(model, fsdp_cfg)
    else:
        print("Warning: CUDA not available, skipping FSDP wrapping.")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.eps,
    )

    # Scheduler
    # Simple constant scheduler for now as per config
    scheduler = None # Implement if needed

    # Dataset
    tokenizer = model.module.tokenizer if isinstance(model, torch.nn.parallel.DistributedDataParallel) or hasattr(model, "module") else model.tokenizer

    # We need to access tokenizer from the wrapped model.
    # FSDP wraps the module.
    # But FSDP flattens parameters. Attributes might not be directly accessible unless using getattr on module.
    # Accessing tokenizer: The tokenizer is not a parameter, so it might be on the underlying module.
    # However, FSDP usually proxies attribute access if not conflicting.

    # If FSDP doesn't proxy:
    # model.tokenizer might fail if it's not exposed.
    # But MLLModel has self.tokenizer.

    # Wait, we need to be careful. The tokenizer is loaded inside MLLModel.
    # It's better to instantiate tokenizer separately or extract it.
    # MLLModel loads it from 'llm.checkpoint'.

    # Let's try accessing it. If it fails, we load it again.
    try:
        tokenizer = model.tokenizer
    except AttributeError:
        # If wrapped, access module
        tokenizer = model.module.tokenizer

    dataset = BitDanceDataset(tokenizer, config, num_samples=1000) # Dummy 1000 samples

    collator = BitDanceCollator(tokenizer, config)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.get("batch_size", 1), # Default to 1 if not in config
        sampler=sampler,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # Training Loop
    total_steps = config.training.total_steps
    start_step = 0

    # Resume logic could go here (using FSDPCheckpoint)

    model.train()

    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step, disable=rank != 0)

    # Infinite iterator over dataloader
    train_iter = iter(dataloader)

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            if sampler:
                sampler.set_epoch(step // len(dataloader)) # Rough approximation
            train_iter = iter(dataloader)
            batch = next(train_iter)

        # Move batch to device
        # Batch is a dict of tensors (mostly)
        # Note: 'vit_image_tensors' is a list of tensors.
        # 'gen_vit_latent_shapes' is list of tuples.

        # MLLModel.forward_train expects specific args.
        # We can unpack the batch dict as kwargs.

        # We need to move tensors to device.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                batch[k] = [x.to(device) for x in v]

        optimizer.zero_grad()

        # Forward
        outputs = model(**batch)

        loss_text = outputs['ce_loss_text'].mean() if outputs['ce_loss_text'] is not None else 0.0
        loss_vision = outputs['ce_loss_vision'].mean() if outputs['ce_loss_vision'] is not None else 0.0

        # Weights
        w_text = config.training.loss_weight_text
        w_vision = config.training.loss_weight_vision

        loss = w_text * loss_text + w_vision * loss_vision

        # Backward
        loss.backward()

        # Clip grad
        if config.training.max_grad_norm > 0:
            model.clip_grad_norm_(config.training.max_grad_norm)

        optimizer.step()

        # Log
        if rank == 0 and step % config.training.log_every == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/loss_text": loss_text.item() if isinstance(loss_text, torch.Tensor) else loss_text,
                "train/loss_vision": loss_vision.item() if isinstance(loss_vision, torch.Tensor) else loss_vision,
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/step": step,
            })
            pbar.set_description(f"Loss: {loss.item():.4f}")

        # Save
        if step > 0 and step % config.training.save_every == 0:
            if torch.cuda.is_available():
                FSDPCheckpoint.fsdp_save_ckpt(
                    ckpt_dir=config.training.results_dir,
                    train_steps=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    data_status=None,
                    logger=wandb if rank == 0 else None, # Pass wandb as logger or a proper logger
                    fsdp_config=fsdp_cfg
                )
            elif rank == 0:
                save_path = os.path.join(config.training.results_dir, f"{step:07d}")
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

        step += 1
        pbar.update(1)

    pbar.close()
    cleanup_distributed()

if __name__ == "__main__":
    main()

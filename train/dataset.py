import torch
from torch.utils.data import Dataset
import numpy as np

class BitDanceDataset(Dataset):
    def __init__(self, tokenizer, config, num_samples=100):
        self.tokenizer = tokenizer
        self.config = config
        self.num_samples = num_samples
        self.patch_size = config.model.vit_patch_size
        self.image_size = (256, 256) # Fixed for now, can be dynamic

        # Pre-calculate tokens
        self.im_start_id = tokenizer.im_start_id
        self.im_end_id = tokenizer.im_end_id
        self.vision_start_id = tokenizer.start_of_image_id
        # self.vision_end_id = tokenizer.end_of_image_id

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate dummy text
        prompt = f"A random image number {idx}"

        # Tokenize text
        # Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = self.tokenizer.encode(text)

        # Generate dummy image
        # shape: [1, 3, H, W]
        image = torch.randn(1, 3, *self.image_size)

        return {
            "text_ids": text_ids,
            "image": image,
            "image_size": self.image_size
        }

class BitDanceCollator:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.patch_size = config.model.vit_patch_size
        self.parallel_num = config.model.head.vision_pred.get("parallel_num", 1)

    def __call__(self, batch):
        vit_image_tensors = []

        packed_text_ids_list = []
        packed_text_indexes_list = []
        packed_vit_token_indexes_list = []
        packed_position_ids_list = []
        sample_lens_list = []

        packed_label_ids_list = []
        ce_loss_indexes_text_list = []

        ce_loss_indexes_vision_list = []
        packed_label_indexes_vision_list = [] # Indices into the flattened vision latents

        gen_vit_latent_shapes = []

        current_seq_offset = 0
        current_vision_latent_offset = 0

        for i, sample in enumerate(batch):
            text_ids = sample["text_ids"]
            image = sample["image"]
            H, W = sample["image_size"]

            vit_image_tensors.append(image)

            # Vision prefix tokens
            h_idx = H // self.patch_size
            w_idx = W // self.patch_size

            res_h_id = getattr(self.tokenizer, f"res_{h_idx}_id")
            if res_h_id is None: res_h_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0

            res_w_id = getattr(self.tokenizer, f"res_{w_idx}_id")
            if res_w_id is None: res_w_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0

            start_of_image_id = self.tokenizer.start_of_image_id
            if start_of_image_id is None: start_of_image_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0

            vision_prefix_ids = [start_of_image_id, res_h_id, res_w_id]

            # Query tokens for parallel decoding
            if self.parallel_num > 1:
                for j in range(1, self.parallel_num):
                    query_id = getattr(self.tokenizer, f"query_{j}_id")
                    if query_id is None: query_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                    vision_prefix_ids.append(query_id)

            # Combined text + vision prefix
            full_text_ids = text_ids + vision_prefix_ids
            len_text = len(full_text_ids)

            # Calculate vision tokens length
            # Note: mllm.py uses 'ps' (patch size in latent space?)
            # In mllm.py:
            # self.ps = int(self.parallel_num ** 0.5)
            # encoded latents: (h * w * p1 * p2) -> (h * w) * (p1 * p2) ?
            # Wait, vt_forward output shape:
            # q = rearrange(quant[b], "c (h p1) (w p2) -> (h w p1 p2) c", p1=ps, p2=ps)
            # So num_tokens = (H_latent * W_latent)

            # Encoder config: z_channels=32, ch_mult=[1,1,2,2,4] -> downsample factor = 2^(5-1) = 16?
            # self.vision_encoder = VQModel(**encoder_config.params)
            # In autoencoder.py: self.num_blocks = len(ch_mult)
            # downsample happens num_blocks-1 times?
            # ch_mult=[1,1,2,2,4] len=5.
            # loop i_level < num_blocks - 1: downsample.
            # So 4 downsamples. 2^4 = 16.
            # So latent map size = H/16, W/16.

            # If patch_size=16 (vit_patch_size), then H/16 is exactly h_idx.
            # However, ps (parallel scale) affects the rearrangement.
            # ps = sqrt(parallel_num).
            # vt_forward rearranges (h p1) (w p2) -> (h w p1 p2).
            # So total tokens = (H/16) * (W/16).
            # But wait, is H/16 the size BEFORE or AFTER rearrangement?
            # If encoder outputs H/16, and we consider ps.
            # Let's assume input H=256, patch_size=16. H/16 = 16.
            # Encoder output is 16x16.
            # rearrange: "c (h p1) (w p2) -> (h w p1 p2) c"
            # If ps=1 (parallel_num=1), then (h) (w) -> (h w). Tokens = 16*16 = 256.
            # If ps=8 (parallel_num=64), then (h*8) (w*8) -> ...
            # Wait, the encoder output resolution depends on the input image resolution.
            # If input is 256x256, encoder output is 16x16 (factor 16).
            # Then rearrange: `rearrange(quant[b], "c (h p1) (w p2) -> (h w p1 p2) c", p1=ps, p2=ps)`
            # This implies `h_enc = h * ps`.
            # So `h = h_enc / ps`.
            # This means the encoder output resolution MUST be divisible by `ps`.

            ps = int(self.parallel_num ** 0.5)
            h_enc = h_idx # 256/16 = 16
            w_enc = w_idx

            if h_enc % ps != 0 or w_enc % ps != 0:
                 raise ValueError(f"Encoder output size ({h_enc}, {w_enc}) must be divisible by ps={ps}")

            num_vision_tokens = h_enc * w_enc

            total_len = len_text + num_vision_tokens
            sample_lens_list.append(total_len)

            # Indices
            packed_text_ids_list.extend(full_text_ids)
            packed_text_indexes_list.extend(range(current_seq_offset, current_seq_offset + len_text))

            packed_vit_token_indexes_list.extend(range(current_seq_offset + len_text, current_seq_offset + total_len))

            packed_position_ids_list.extend(range(total_len))

            # Labels
            # For text, we predict next token.
            # Input: [BOS] user ...
            # Target: user ...
            # We shift inside the model or here?
            # mllm.py:
            # logits_text = self.llm_model.lm_head(llm_embed[ce_loss_indexes_text])
            # ce_loss_text = F.cross_entropy(logits_text, packed_label_ids, reduction="none")

            # So we need to provide labels aligned with `ce_loss_indexes_text`.
            # Typically, we compute loss on tokens starting from [BOS] (predicting next).
            # But usually the last token prediction is matched against label.
            # If `ce_loss_indexes_text` selects the input embeddings at positions t,
            # then `packed_label_ids` should be token at t+1.

            # Let's assume we want to train on the whole text sequence + vision.
            # text_ids: t0, t1, t2, ..., t_last
            # inputs: t0, t1, ...
            # labels: t1, t2, ...

            # Vision labels:
            # We want to predict vision tokens.
            # Inputs: text + vision_prefix.
            # Prediction: first vision token?
            # The model is autoregressive over vision tokens too.
            # Input: [text] [v0] -> predict [v1]?
            # Wait, `diffusion_parallel_x` head might be different.

            # In `forward_train`:
            # if "diffusion_parallel" in self.vision_head_type:
            #     embed_loss = llm_embed[ce_loss_indexes_vision] + pos_embed_for_diff
            #     ce_loss_vision = self.vision_diffusion_head(packed_labels_vision, embed_loss)

            # `packed_labels_vision` are the ground truth latents.
            # `ce_loss_indexes_vision` selects the inputs to the head.
            # If we are predicting v0, v1, ...
            # We input [text+prefix] -> predict v0?
            # Or [text+prefix+v0] -> predict v1?

            # For diffusion head, typically it takes the conditioning embedding and the noisy target.
            # Here `embed_loss` seems to be the conditioning.
            # `llm_embed[ce_loss_indexes_vision]` corresponds to the output of the LLM at the positions WHERE we want to generate vision tokens.
            # So if we have `[text] [prefix] [v0] [v1] ...`
            # To predict `v0`, we need the embedding at `[prefix]`.
            # To predict `v1`, we need embedding at `[v0]`.

            # So `ce_loss_indexes_vision` should mark the positions that PRECEDE the vision tokens we want to predict.
            # The tokens at these positions are: `[text] ... [vision_prefix] [v0] [v1] ... [v_last-1]`.
            # The targets are `[v0] [v1] ... [v_last]`.

            # The indices in `packed_sequence`:
            # Text part: 0 .. len_text-1
            # Vision part: len_text .. total_len-1

            # Vision tokens are at `len_text .. total_len-1`.
            # So `ce_loss_indexes_vision` should cover `len_text-1` (last prefix token) to `total_len-2`.
            # This corresponds to inputs that predict `v0` .. `v_last-1`.
            # Wait, usually we predict ALL vision tokens.
            # If there are N vision tokens.
            # We need N predictions.
            # So we need N input positions.
            # The input positions are `len_text-1` (the last token of prefix) + `len_text` to `total_len-2` (the vision tokens except the last one).
            # This gives N positions?
            # 0-indexed:
            # prefix: ends at L-1.
            # vision: L, L+1, ..., L+N-1.
            # Total len = L+N.
            # Inputs: ... [L-1], [L], ..., [L+N-2]. (N tokens)
            # Targets: [L], [L+1], ..., [L+N-1]. (N tokens)

            # Wait, `packed_vit_token_indexes` puts the vision embeddings at `len_text` to `total_len-1`.
            # These are the GROUND TRUTH vision embeddings (from encoder).
            # So at index `L` we have `v0`.
            # If we want to predict `v0`, we should use the embedding from `L-1`.
            # So `ce_loss_indexes_vision` should be indices `L-1` to `L+N-2`.

            # BUT, the `packed_sequence` is constructed with:
            # Text at indices `packed_text_indexes`
            # Vision at indices `packed_vit_token_indexes`
            # So `L-1` is the last text/prefix token.
            # `L` is `v0`.
            # `L+N-1` is `v_last`.

            # So correct.

            # Text Loss:
            # Predict text tokens.
            # Inputs: `0` to `len_text-2`.
            # Targets: `text_ids[1:]`.
            # `ce_loss_indexes_text` should be `0` to `len_text-2`.

            # Let's populate the lists.

            # Text loss
            # We predict all text tokens except the first one (BOS), based on previous tokens.
            # Also we might want to mask out user prompt if we only train assistant.
            # But for now let's train on everything except BOS.

            start_text_idx = current_seq_offset
            end_text_idx = current_seq_offset + len_text - 1 # Last text token index

            # Inputs: 0 .. len_text-2 (predicts 1 .. len_text-1)
            # Labels: 1 .. len_text-1
            # Note: last text token (prefix end) predicts first vision token.

            # We split text and vision loss.

            # Text tokens to predict: `full_text_ids[1:]`.
            # Corresponding inputs: `full_text_ids[:-1]`.
            # Indices: `start_text_idx` to `end_text_idx - 1`.

            ce_loss_indexes_text_list.extend(range(start_text_idx, end_text_idx))
            packed_label_ids_list.extend(full_text_ids[1:])

            # Vision loss
            # We predict `num_vision_tokens`.
            # Inputs indices: `end_text_idx` (last text token) to `end_text_idx + num_vision_tokens - 1`.
            # Targets indices in latents: 0 to num_vision_tokens-1.

            start_vision_input_idx = end_text_idx
            end_vision_input_idx = end_text_idx + num_vision_tokens

            ce_loss_indexes_vision_list.extend(range(start_vision_input_idx, end_vision_input_idx))

            # packed_label_indexes_vision selects which latents are targets.
            # The latents are concatenated for the batch.
            # Current sample has `num_vision_tokens` latents.
            # They are at `current_vision_latent_offset` to `... + num_vision_tokens`.

            packed_label_indexes_vision_list.extend(range(current_vision_latent_offset, current_vision_latent_offset + num_vision_tokens))

            gen_vit_latent_shapes.append((H, W))

            # Update offsets
            current_seq_offset += total_len
            current_vision_latent_offset += num_vision_tokens

        # Convert to tensors
        # ...

        sequence_length = current_seq_offset

        return {
            "vit_image_tensors": vit_image_tensors,
            "vit_token_indexes_for_encoder": None, # Not used in code?
            "packed_vit_rope_coords": None, # Not used
            "vit_token_seqlens": None, # Not used
            "vit_latent_shapes": None, # Not used
            "gen_vit_latent_shapes": gen_vit_latent_shapes,
            "sequence_length": sequence_length,
            "sample_lens": torch.tensor(sample_lens_list, dtype=torch.int32),
            "packed_position_ids": torch.tensor(packed_position_ids_list, dtype=torch.long),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes_list, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids_list, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes_list, dtype=torch.long),
            "ce_loss_indexes_text": self._indices_to_bool(ce_loss_indexes_text_list, sequence_length),
            "packed_label_ids": torch.tensor(packed_label_ids_list, dtype=torch.long),
            "ce_loss_indexes_vision": self._indices_to_bool(ce_loss_indexes_vision_list, sequence_length),
            "packed_label_indexes_vision": torch.tensor(packed_label_indexes_vision_list, dtype=torch.long),
        }

    def _indices_to_bool(self, indices, length):
        mask = torch.zeros(length, dtype=torch.bool)
        mask[indices] = True
        return mask

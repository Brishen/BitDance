import torch
from torch import nn
from typing import Optional, Union
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
from einops import rearrange

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    apply_rotary_pos_emb,
    CausalLMOutputWithPast,
    Qwen2Config
)
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

logger = logging.get_logger(__name__)

class Qwen2PackedAttention(Qwen2Attention):
    """
    Qwen2 Attention module modified to support packed sequences for efficient training
    and inference with variable-length inputs. This module uses flash_attn_varlen_func.

    It inherits from Qwen2Attention to reuse the projection layers and configuration.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # This will be ignored in packed mode
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sample_lens: Optional[torch.LongTensor] = None, # New argument for packed sequences
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

        # If sample_lens is not provided, it means we are not using packed attention.
        # Fall back to the original Qwen2Attention's forward method.
        # This makes the module versatile for both packed and padded inputs.
        if sample_lens is None:
            # logger.warning_once(
            #     "Qwen2PackedAttention received `sample_lens=None`. Falling back to the standard attention mechanism. "
            #     "This may be slow. For packed attention, ensure `sample_lens` is provided."
            # )
            # Call the parent class's forward method
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs
            )

        # --- Packed Attention Logic ---

        # The input hidden_states is already a packed tensor of shape [total_tokens, hidden_size]
        total_tokens = hidden_states.shape[-2]

        # 1. Project Q, K, V from the packed hidden_states
        # Input: [total_tokens, hidden_size] -> Output: [total_tokens, num_heads, head_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(total_tokens, self.config.num_attention_heads, self.head_dim)
        key_states = key_states.view(total_tokens, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.view(total_tokens, self.config.num_key_value_heads, self.head_dim)

        # 2. Apply Rotary Positional Embeddings (RoPE)
        if position_embeddings is None:
             # Qwen2Attention might compute rotary internally if not passed, but usually it expects them.
             # In standard Qwen2Attention forward, cos/sin are passed or computed.
             # We assume they are passed as in original wrapper.
             pass
        else:
             cos, sin = position_embeddings
             query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos.squeeze(0), sin.squeeze(0))

        # 4. Prepare for Flash Attention Varlen
        # Create cumulative sequence lengths for flash_attn_varlen_func
        cu_seqlens = nn.functional.pad(torch.cumsum(sample_lens, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = sample_lens.max().item()

        # 5. Call Flash Attention Varlen Function
        # Note: flash_attn_varlen_func expects inputs of shape [total_tokens, num_heads, head_dim]
        if flash_attn_varlen_func is None:
            raise ImportError("flash_attn is not installed, but packed attention requires it. Please install flash_attn or provide `sample_lens=None` to disable packed attention.")

        attn_output = flash_attn_varlen_func(
            q=query_states.to(torch.bfloat16),
            k=key_states.to(torch.bfloat16),
            v=value_states.to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=True,
            # The scaling factor is applied inside the kernel
            softmax_scale=self.scaling,
        )

        # 6. Final Projection
        # Reshape and apply the output projection
        attn_output = attn_output.reshape(total_tokens, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if attn_output.ndim == 2:
            attn_output = attn_output.unsqueeze(0)

        # Packed attention does not return attention weights or handle past_key_values
        return attn_output, None, past_key_values


class Qwen2ForCausalLMWrapper(Qwen2ForCausalLM):
    """
    A wrapper for Qwen2ForCausalLM that enables packed attention by monkey-patching
    the attention modules.
    """
    def __init__(self, config: Qwen2Config, use_packed_attn: bool = True, **kwargs):
        # Initialize the original model first
        super().__init__(config, **kwargs)

        self.use_packed_attn = use_packed_attn
        if self.use_packed_attn:
            self._enable_packed_attention()

    def _enable_packed_attention(self):
        """
        Replaces the standard Qwen2Attention modules with our Qwen2PackedAttention module.
        """
        logger.info("Enabling packed attention by replacing Qwen2Attention with Qwen2PackedAttention.")
        for layer in self.model.layers:
            packed_attn_module = Qwen2PackedAttention(config=self.config, layer_idx=layer.self_attn.layer_idx)
            packed_attn_module.load_state_dict(layer.self_attn.state_dict())
            layer.self_attn = packed_attn_module

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # Will be ignored if sample_lens is provided
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        sample_lens: Optional[torch.LongTensor] = None, # New argument
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        is_packed_path = self.use_packed_attn and sample_lens is not None
        if is_packed_path:
            if input_ids is not None and input_ids.shape[0] > 1:
                raise ValueError("For packed attention, `input_ids` must be a packed tensor with batch_size=1.")

            attention_mask = None # Not used in varlen attention
            kwargs["sample_lens"] = sample_lens

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

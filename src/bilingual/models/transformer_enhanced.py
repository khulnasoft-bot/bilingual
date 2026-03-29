"""
Enhanced Transformer model for Bangla-English translation with advanced features.

Features:
- Rotary Positional Embeddings (RoPE)
- SwiGLU activation
- Layer normalization with pre-norm
- Multi-query attention
- Gated residual connections
- Flash Attention 2
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    # Keep the import but don't assign it to a name since it's used in the class
    from flash_attn import flash_attn_func  # noqa: F401

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) from RoFormer."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Will be computed on first forward pass
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self.max_seq_len_cached = 0

    def _update_cos_sin_cache(self, seq_len: int, device=None):
        if (
            seq_len <= self.max_seq_len_cached
            and self.cos_cached is not None
            and self.cos_cached.device == device
        ):
            return

        self.max_seq_len_cached = seq_len
        position = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Compute frequencies on-the-fly
        freqs = torch.einsum("i,j->ij", position, self.inv_freq.to(device))
        # Concatenate for both sin and cos
        emb = torch.cat([freqs, freqs], dim=-1)

        # Register buffers with proper device
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: Tensor, seq_dim: int = -2) -> Tensor:
        """Applies rotary embeddings to input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim] or
                [seq_len, batch_size, num_heads, head_dim]
            seq_dim: Dimension containing the sequence length (default: -2)

        Returns:
            Tensor with rotary position embeddings applied
        """
        seq_len = x.size(seq_dim)

        # Update cache if needed
        self._update_cos_sin_cache(seq_len, device=x.device)

        # Reshape for broadcasting if needed
        if seq_dim != -2:
            x = x.transpose(1, seq_dim)

        # Get the first dim elements for rotary embedding
        x_rot = x[..., : self.dim]
        x_pass = x[..., self.dim :]

        # Apply rotary embeddings (in-place for memory efficiency)
        x_rot = (x_rot * self.cos_cached[:, :, :seq_len]) + (
            self._rotate_half(x_rot) * self.sin_cached[:, :, :seq_len]
        )

        # Concatenate with the rest of the dimensions
        x = torch.cat([x_rot, x_pass], dim=-1)

        # Restore original shape if needed
        if seq_dim != -2:
            x = x.transpose(1, seq_dim)

        return x

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotates half the hidden dims of the input.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Rotated tensor of same shape as input
        """
        half_dim = x.shape[-1] // 2
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU activation function from https://arxiv.org/abs/2002.05202"""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize weights similar to GPT-2
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
            nn.init.zeros_(self.w2.bias)
            nn.init.zeros_(self.w3.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MultiQueryAttention(nn.Module):
    """Multi-query attention from https://arxiv.org/abs/1911.02150"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_flash_attn: bool = HAS_FLASH_ATTN,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.use_flash_attn = use_flash_attn

        # Multi-query attention: shared (key, value) across heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize weights similar to GPT-2
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass for multi-query attention."""
        batch_size, tgt_len, _ = query.size()

        # Project queries, keys, values
        q = self.q_proj(query)  # [batch_size, tgt_len, embed_dim]
        k = self.k_proj(key)  # [batch_size, src_len, head_dim]
        v = self.v_proj(value)  # [batch_size, src_len, head_dim]

        # Reshape for multi-head attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch_size, num_heads, tgt_len, head_dim]
        k = k.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )  # [batch_size, num_heads, src_len, head_dim]
        v = v.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )  # [batch_size, num_heads, src_len, head_dim]

        # Scale q
        q = q * self.scaling

        # Compute attention scores
        attn_weights = torch.matmul(
            q, k.transpose(-2, -1)
        )  # [batch_size, num_heads, tgt_len, src_len]

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        # Compute attention probabilities
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, tgt_len, head_dim]

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            return attn_output, attn_probs
        return attn_output, None


class TransformerEncoderLayer(nn.Module):
    """Enhanced Transformer encoder layer with pre-norm and gated residual."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "swiglu",
        layer_norm_eps: float = 1e-5,
        use_flash_attn: bool = HAS_FLASH_ATTN,
    ):
        super().__init__()

        # Self-attention with pre-norm
        self.self_attn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = MultiQueryAttention(
            d_model, nhead, dropout=dropout, use_flash_attn=use_flash_attn
        )

        # Feed-forward with SwiGLU
        self.ffn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = SwiGLU(d_model, dim_feedforward)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Gating mechanism
        self.attn_gate = nn.Parameter(torch.ones(1))
        self.ffn_gate = nn.Parameter(torch.ones(1))

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for the encoder layer."""
        # Self-attention with pre-norm and gated residual
        x = self.self_attn_norm(src)
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout(attn_out) * self.attn_gate

        # Feed-forward with pre-norm and gated residual
        x = self.ffn_norm(src)
        ffn_out = self.ffn(x)
        src = src + self.dropout(ffn_out) * self.ffn_gate

        return src


class TransformerDecoderLayer(nn.Module):
    """Enhanced Transformer decoder layer with pre-norm and gated residual."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "swiglu",
        layer_norm_eps: float = 1e-5,
        use_flash_attn: bool = HAS_FLASH_ATTN,
    ):
        super().__init__()

        # Self-attention with pre-norm
        self.self_attn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = MultiQueryAttention(
            d_model, nhead, dropout=dropout, use_flash_attn=use_flash_attn
        )

        # Cross-attention with pre-norm
        self.cross_attn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attn = MultiQueryAttention(
            d_model, nhead, dropout=dropout, use_flash_attn=use_flash_attn
        )

        # Feed-forward with SwiGLU
        self.ffn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = SwiGLU(d_model, dim_feedforward)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Gating mechanism
        self.self_attn_gate = nn.Parameter(torch.ones(1))
        self.cross_attn_gate = nn.Parameter(torch.ones(1))
        self.ffn_gate = nn.Parameter(torch.ones(1))

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for the decoder layer."""
        # Self-attention with pre-norm and gated residual
        x = self.self_attn_norm(tgt)
        self_attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout(self_attn_out) * self.self_attn_gate

        # Cross-attention with pre-norm and gated residual
        x = self.cross_attn_norm(tgt)
        cross_attn_out, _ = self.cross_attn(
            x,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout(cross_attn_out) * self.cross_attn_gate

        # Feed-forward with pre-norm and gated residual
        x = self.ffn_norm(tgt)
        ffn_out = self.ffn(x)
        tgt = tgt + self.dropout(ffn_out) * self.ffn_gate

        return tgt


class EnhancedTransformer(nn.Module):
    """Enhanced Transformer model with modern architectural improvements."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
        pad_idx: int = 0,
        layer_norm_eps: float = 1e-5,
        use_rotary_emb: bool = True,
        use_flash_attn: bool = HAS_FLASH_ATTN,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_seq_length = max_seq_length

        # Token embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # Positional embeddings
        self.pos_emb = (
            RotaryPositionalEmbedding(d_model // nhead, max_seq_length) if use_rotary_emb else None
        )

        # Encoder
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Decoder
        self.decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Output layer
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Share weights between input and output embeddings
        self.output_proj.weight = self.tgt_tok_emb.weight

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize embeddings
        nn.init.normal_(self.src_tok_emb.weight, mean=0, std=self.d_model**-0.5)
        nn.init.normal_(self.tgt_tok_emb.weight, mean=0, std=self.d_model**-0.5)

        # Initialize output projection
        if hasattr(self.output_proj, "bias") and self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.0)

        # Initialize layer norms
        for layer in self.encoder:
            if hasattr(layer, "self_attn_norm"):
                nn.init.constant_(layer.self_attn_norm.weight, 1.0)
                nn.init.constant_(layer.self_attn_norm.bias, 0.0)
            if hasattr(layer, "ffn_norm"):
                nn.init.constant_(layer.ffn_norm.weight, 1.0)
                nn.init.constant_(layer.ffn_norm.bias, 0.0)

        for layer in self.decoder:
            if hasattr(layer, "self_attn_norm"):
                nn.init.constant_(layer.self_attn_norm.weight, 1.0)
                nn.init.constant_(layer.self_attn_norm.bias, 0.0)
            if hasattr(layer, "cross_attn_norm"):
                nn.init.constant_(layer.cross_attn_norm.weight, 1.0)
                nn.init.constant_(layer.cross_attn_norm.bias, 0.0)
            if hasattr(layer, "ffn_norm"):
                nn.init.constant_(layer.ffn_norm.weight, 1.0)
                nn.init.constant_(layer.ffn_norm.bias, 0.0)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the enhanced transformer model."""
        # Source and target embeddings
        src_emb = self.src_tok_emb(src) * (self.d_model**0.5)
        tgt_emb = self.tgt_tok_emb(tgt) * (self.d_model**0.5)

        # Apply positional embeddings if using them
        if self.pos_emb is not None:
            src_emb = self.pos_emb(src_emb)
            tgt_emb = self.pos_emb(tgt_emb)

        # Encode source
        memory = src_emb
        for layer in self.encoder:
            memory = layer(
                memory,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        # Decode
        output = tgt_emb
        for layer in self.decoder:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # Project to vocabulary
        output = self.output_proj(output)

        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq: Tensor) -> Tensor:
        """Create padding mask for sequences."""
        return seq == self.pad_idx


# Example usage:
# model = EnhancedTransformer(
#     src_vocab_size=32000,
#     tgt_vocab_size=32000,
#     d_model=512,
#     nhead=8,
#     num_encoder_layers=6,
#     num_decoder_layers=6,
#     dim_feedforward=2048,
#     dropout=0.1,
#     max_seq_length=1024,
# )

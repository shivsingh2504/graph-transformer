from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from model.layers import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    TokenEmbedding,
)

_D_MODEL: int = 128
_N_HEADS: int = 4
_N_LAYERS: int = 3
_D_FF: int = 512
_DROPOUT: float = 0.1


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = _D_MODEL,
        n_heads: int = _N_HEADS,
        d_ff: int = _D_FF,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = _D_MODEL,
        n_heads: int = _N_HEADS,
        d_ff: int = _D_FF,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=self_attn_mask)))
        x = self.norm2(x + self.dropout(
            self.cross_attn(x, encoder_output, encoder_output, mask=cross_attn_mask)
        ))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int = _N_LAYERS,
        d_model: int = _D_MODEL,
        n_heads: int = _N_HEADS,
        d_ff: int = _D_FF,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size, d_model=d_model, dropout=dropout
        )
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(src_ids)
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int = _N_LAYERS,
        d_model: int = _D_MODEL,
        n_heads: int = _N_HEADS,
        d_ff: int = _D_FF,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size, d_model=d_model, dropout=dropout
        )
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        tgt_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(tgt_ids)
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int = _N_LAYERS,
        d_model: int = _D_MODEL,
        n_heads: int = _N_HEADS,
        d_ff: int = _D_FF,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

    def encode(
        self,
        src_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(src_ids, src_mask)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        decoder_out = self.decoder(tgt_ids, encoder_output, self_attn_mask, cross_attn_mask)
        return self.output_projection(decoder_out)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_self_attn_mask: Optional[torch.Tensor] = None,
        tgt_cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_output = self.encode(src_ids, src_mask)
        return self.decode(tgt_ids, encoder_output, tgt_self_attn_mask, tgt_cross_attn_mask)
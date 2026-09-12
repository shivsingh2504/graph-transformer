from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from src.model.layers import(MultiHeadAttention,TokenEmbedding,PositionwiseFeedForward)

_D_MODEL : int = 128
_N_HEADS : int  = 4
_N_LAYERS : int  = 3
_D_FF : int = 512
_DROPOUT : float = 0.1

class EncoderLayer(nn.Module):
  def __init__(
    self,
    d_model : int = _D_MODEL,
    n_heads : int = _N_HEADS,
    d_ff : int = _D_FF,
    dropout : float = _DROPOUT
  ):
    super().__init__()
    self.self_attn = MultiHeadAttention(d_model=d_model,dropout=dropout,n_heads=n_heads)
    self.ffn = PositionwiseFeedForward(dropout=dropout,d_model=d_model,d_ff=d_ff)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(p=dropout)
    def forward(
      self,
      x : torch.Tensor,
      encoder_output : torch.Tensor,
      self_attn_mask : Optional[torch.Tensor] = None,
      cross_attn_mask : Optional[torch.Tensor] = None
    ):
      x = self.norm1(x + self.dropout(self.self_attn(x,x,x,mask=self_attn_mask)))
      x = self.norm2(x + self.dropout(self.cross_attn(x,encoder_output,encoder_output,mask = cross_attn_mask)))
      x = self.norm3(x + self.dropout(self.ffn(x)))
      return x
class DecoderLayer(nn.Module):
  def __init__(
    self,
    d_model:int = _D_MODEL,
    n_heads : int = _N_HEADS,
    d_ff : int = _D_FF,
    droput :float = _DROPOUT
  )->None:
    super().__init__()
    self.self_attn = MultiHeadAttention(d_model=d_model,n_heads=n_heads,dropout=droput)
    self.cross_attn = MultiHeadAttention(d_model=d_model,n_heads=n_heads,dropout=droput)
    self.ffn = PositionwiseFeedForward(d_model=d_model,d_ff=d_ff,dropout=droput)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(p=droput)
  def forward(
    self,
    x:torch.Tensor,
    encoder_output : torch.Tensor,
    self_attn_mask : Optional[torch.Tensor]=None,
    cross_attn_mask : Optional[torch.Tensor]=None,
  )->torch.Tensor:
    x = self.norm1(x+self.dropout(self.self_attn(x,x,x,mask = self_attn_mask)))
    x = self.norm2(x + self.dropout(
      self.cross_attn(x,encoder_output,encoder_output,mask = cross_attn_mask)
    ))
    x = self.norm3(x + self.dropout(self.ffn(x)))
    return x
  
class Encoder(nn.Module):
  def __init__(
    self,
    vocab_size : int,
    n_layers : int = _N_LAYERS,
    n_heads : int = _N_HEADS,
    d_model : int = _D_MODEL,
    d_ff : int = _D_FF,
    dropout : float = _DROPOUT
  )->None:
    super().__init__()
    self.embedding = TokenEmbedding(
      vocab_size=vocab_size,d_model=d_model,dropout=dropout
    )
    self.layers = nn.ModuleList([
      EncoderLayer(d_model=d_model,n_heads=n_heads,d_ff=d_ff,dropout=dropout)
      for _ in range(n_layers)
    ])
  def forward(
    self,
    src_ids : torch.Tensor,
    src_mask : Optional[torch.Tensor] = None,
  )-> torch.Tensor:
    x = self.embedding(src_ids)
    for layer  in self.layers:
      x = layer(x,mask= src_mask)
    return x

class Decoder(nn.Module):
  def __init__(
    self,
    vocab_size  :int,
    n_layers : int = _N_LAYERS,
    n_heads  : int = _N_HEADS,
    d_ff : int = _D_FF,
    d_model  : int = _D_MODEL,
    dropout : float = _DROPOUT
  )->None:
    super().__init__()
    self.embedding = TokenEmbedding(
      vocab_size=vocab_size,dropout=dropout,d_model=d_model
    )
    self.layers = nn.ModuleList([
      DecoderLayer(d_model=d_model,d_ff=d_ff,droput=dropout,n_heads=n_heads)
      for _ in range(n_layers)
    ])
  def forward(
    self,
    tgt_ids : torch.Tensor,
    encoder_output : torch.Tensor,
    self_attn_mask : Optional[torch.Tensor]=None,
    cross_attn_mask : Optional[torch.Tensor] = None
  )->torch.Tensor:
    x = self.embedding(tgt_ids)
    for layer in self.layers:
      x = layer(x,encoder_output,self_attn_mask,cross_attn_mask)
    return x


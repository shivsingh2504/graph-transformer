from __future__ import annotations
import math
from typing import Optional,List,Tuple,Union
import torch 
import torch.nn as nn
import torch.nn.functional as F

_D_MODEL: int = 128
_N_HEADS: int = 4
_D_FF: int = 512
def create_padding_mask(seq:torch.Tensor,pad_token_id:int)->torch.Tensor:
  mask = (seq != pad_token_id)
  return mask.unsqueeze(1).unsqueeze(2)
def create_causal_mask(seq_len:int,device:torch.device = torch.device("cpu"))->torch.Tensor:
  mask = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=device))
  return mask.unsqueeze(0).unsqueeze(0)

class PositionalEncoding(nn.Module):
  def __init__(
    self,
    d_model:int = _D_MODEL,
    max_len:int = 5000,
    dropout : float = 0.0
    
  )->None:
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(max_len,d_model)
    position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0,d_model,2,dtype = torch.float)
      * (-math.log(10000.0)/d_model)
    )
    pe[:,0::2] = torch.sin(position*div_term)
    pe[:,1::2] = torch.cos(position*div_term)
    self.register_buffer("pe",pe.unsqueeze(0))
  def forward (self,x:torch.Tensor)->torch.Tensor:
    seq_len = x.size(1)
    x = x + self.pe[:, :seq_len, :]
    return self.dropout(x)

class MultiHeadAttention(nn.Module):
  def __init__(
    self,
    d_model:int = _D_MODEL,
    n_heads: int = _N_HEADS,
    dropout : float = 0.0
  )->None:
    super().__init__()
    assert d_model % n_heads == 0, (
      f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    )
    self.d_model = d_model
    self.n_heads = n_heads
    self.d_head = d_model // n_heads
    self.scale = math.sqrt(self.d_head)
    self.dropout = nn.Dropout(p=dropout)
    
    self.W_q = nn.Linear(d_model, d_model, bias=False)
    self.W_k = nn.Linear(d_model, d_model, bias=False)
    self.W_v = nn.Linear(d_model, d_model, bias=False)
    self.W_o = nn.Linear(d_model, d_model, bias=False)
    
  def _split_heads(self,x:torch.Tensor)->torch.Tensor:
    batch,seq,_ = x.shape
    return (
      x.view(batch, seq, self.n_heads, self.d_head)
      .transpose(1, 2)
    )
  
  def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
    batch, _, seq, _ = x.shape
    return (
      x.transpose(1, 2)
      .contiguous()
      .view(batch, seq, self.d_model)
    )
  def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    Q = self._split_heads(self.W_q(query))
    K = self._split_heads(self.W_k(key))
    V = self._split_heads(self.W_v(value))

    scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

    if mask is not None:
      scores = scores.masked_fill(~mask, -1e9)

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights_dropped = self.dropout(attn_weights)

    out = torch.matmul(attn_weights_dropped, V)
    out = self.W_o(self._merge_heads(out))

    if return_attn_weights:
      return out, attn_weights
    return out
  
  
class PositionwiseFeedForward(nn.Module):
  def __init__(
    self,
    d_model: int = _D_MODEL,
    d_ff: int = _D_FF,
    dropout: float = 0.0,
  ) -> None:
      super().__init__()
      self.linear1 = nn.Linear(d_model, d_ff)
      self.linear2 = nn.Linear(d_ff, d_model)
      self.dropout = nn.Dropout(p=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.linear2(self.dropout(F.relu(self.linear1(x))))
    
class TokenEmbedding(nn.Module):
  def __init__(
    self,
    vocab_size: int,
    d_model: int = _D_MODEL,
    max_len: int = 5000,
    dropout: float = 0.0,
  ) -> None:
      super().__init__()
      self.d_model = d_model
      self.scale = math.sqrt(d_model)
      self.embedding = nn.Embedding(vocab_size, d_model)
      self.pos_encoding = PositionalEncoding(
        d_model=d_model, max_len=max_len, dropout=dropout
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      embedded = self.embedding(x) * self.scale
      return self.pos_encoding(embedded)
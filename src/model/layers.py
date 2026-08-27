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
def create_casual_mask(seq_len:int,device:torch.device = torch.device("cpu"))->torch.Tensor:
  mask = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=device))
  return mask.unsqueeze(0).unsqueeze(0)

class PositionalEncoding(nn.Module):
  def __init__(
    self,
    d_model:int = _D_MODEL,
    max_len:int = 5000,
    dropout : float = 0.1
    
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

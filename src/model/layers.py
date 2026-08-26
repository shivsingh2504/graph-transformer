from __future__ import annotations
import math
from typing import Optional,List,Tuple,Union
import torch 
import torch.nn as nn
import torch.nn.functional as F

def create_padding_mask(seq:torch.Tensor,pad_token_id:int)->torch.Tensor:
  mask = (seq != pad_token_id)
  return mask.unsqueeze(1).unsqueeze(2)
def create_casual_mask(seq_len:int,device:torch.device = torch.device("cpu"))->torch.Tensor:
  mask = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=device))
  return mask.unsqueeze(0).unsqueeze(0)

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

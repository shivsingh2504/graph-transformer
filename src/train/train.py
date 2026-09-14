from __future__ import annotations
import os
import sys
from typing import Tuple,List,Dict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader,Dataset

from data.dataset_generator import DatasetSplit
from data.tokenizer import GraphTokenizer
from model.layers import create_causal_mask,create_padding_mask
from model.model import Transformer


_D_MODEL : int = 128
_N_HEADS : int = 4
_N_LAYERS: int = 3
_D_FF:int = 512
_DROPOUT : float = 0.1
_LR: float = 1e-4
_WEIGHT_DECAY : float = 1e-2
_WARMUP_STEPS:int = 400
_BATCH_SIZE: int = 32
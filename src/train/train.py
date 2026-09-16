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

def _linear_warmup_schedule(step:int,warmup_steps:int)->float:
  if warmup_steps <= 0:
    return 1.0
  if step < warmup_steps:
    return float(step+1) / float(warmup_steps)
  return 1.0

class GraphPathDataset(Dataset):
  def __init__(self,split: DatasetSplit, tokenizer:GraphTokenizer)->None:
    self.tokenizer = tokenizer
    self.examples = List[Tuple[List[int],List[int]]] = [
      (tokenizer.encode_graph(graph), tokenizer.encode_path(path))
      for graph , path in split.examples
    ]
  
  def __len__(self)->int:
    return len(self.examples)

  def __getitem__(self, idx:int)->Tuple[List[int],List[int]]:
    return self.examples[idx]
  
def collate_fn(
  batch: List[Tuple[List[int],List[int]]],
  pad_id:int,
)->Tuple[torch.Tensor,torch.Tensor]:
  src_seqs , tgt_seqs = zip(*batch)
  max_src = max(len(s) for s in src_seqs)
  max_tgt = max(len(t) for t in tgt_seqs)
  def _pad(seqs: Tuple[List[int],...],max_len:int)->torch.Tensor:
    return torch.Tensor(
      [s+[pad_id]*(max_len - len(s)) for s in seqs],
      dtype = torch.long,
    )
  return _pad(src_seqs,max_src), _pad(tgt_seqs,max_tgt)

def make_dataloader(
  split:DatasetSplit,
  tokenizer:GraphTokenizer,
  batch_size:int = _BATCH_SIZE,
  shuffle : bool = True,
)->DataLoader:
  dataset = GraphPathDataset(split,tokenizer)
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle = shuffle,
    collate_fn = lambda batch: collate_fn(batch,tokenizer.pad_token_id),
  )

def _make_masks(
  src :torch.Tensor,
  decoder_input:torch.Tensor,
  pad_id : int,
)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
  device = src.device
  tgt_len = decoder_input.size(1)
  src_mask = create_padding_mask(src,pad_id)
  tgt_pad_mask = create_padding_mask(decoder_input,pad_id)
  causal_mask = create_causal_mask(tgt_len,device=device)
  tgt_self_attn_mask = causal_mask & tgt_pad_mask
  tgt_cross_attn_mask = src_mask
  return src_mask,tgt_self_attn_mask,tgt_cross_attn_mask

def _train_epoch(
  model:Transformer,
  loader:DataLoader,
  optimizer: torch.optim.Optimizer,
  criterion: nn.CrossEntropyLoss,
  device: torch.device,
  scheduler: LambdaLR | None = None
)->float:
  model.train()
  total_loss = 0.0,
  n_batches = 0
  
  for src, tgt in loader:
    src = src.to(device)
    tgt = tgt.to(device)
    decoder_input = tgt[:, :-1]
    target_output = tgt[:, 1:]
    src_mask,tgt_self_attn_mask,tgt_cross_attn_mask = _make_masks(src,decoder_input,criterion.ignore_index)
    optimizer.zero_grad()
    logits = model(
      src,
      decoder_input,
      src_mask=src_mask,
      tgt_self_attn_mask=tgt_self_attn_mask,
      tgt_cross_attn_mask=tgt_cross_attn_mask
    )
    loss = criterion(
      logits.reshape(-1,logits.size(-1)),
      target_output.reshape(-1),
    )
    loss.backward()
    optimizer.step()
    if scheduler is not None:
      scheduler.step()
    total_loss += loss.item()
    n_batches += 1
  return total_loss / n_batches if n_batches > 0 else 0.0

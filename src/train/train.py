from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from data.dataset_generator import DatasetSplit
from data.tokenizer import GraphTokenizer
from model.layers import create_causal_mask, create_padding_mask
from model.model import Transformer

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
_D_MODEL: int = 128
_N_HEADS: int = 4
_N_LAYERS: int = 3
_D_FF: int = 512
_DROPOUT: float = 0.0
_LR: float = 5e-4
_WEIGHT_DECAY: float = 1e-2
_WARMUP_STEPS: int = 4000
_BATCH_SIZE: int = 32


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _linear_warmup_schedule(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    return 1.0


def _warmup_and_decay_schedule(
    step: int,
    warmup_steps: int,
    decay_start_step: int,
    total_steps: int,
) -> float:
    """
    Combined warmup + decay schedule.
    - Steps 0 to warmup_steps-1: linear warmup from 0 to 1
    - Steps warmup_steps to decay_start_step-1: hold at 1.0
    - Steps decay_start_step to total_steps-1: linear decay from 1.0 to 0.0
    """
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    elif step < decay_start_step:
        return 1.0
    else:
        decay_steps = total_steps - decay_start_step
        if decay_steps <= 0:
            return 0.0
        progress = (step - decay_start_step) / decay_steps
        return max(0.0, 1.0 - progress)


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class GraphPathDataset(Dataset):
    def __init__(self, split: DatasetSplit, tokenizer: GraphTokenizer) -> None:
        self.tokenizer = tokenizer
        self.examples: List[Tuple[List[int], List[int]]] = [
            (tokenizer.encode_graph(graph), tokenizer.encode_path(path))
            for graph, path in split.examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.examples[idx]


def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    src_seqs, tgt_seqs = zip(*batch)

    max_src = max(len(s) for s in src_seqs)
    max_tgt = max(len(t) for t in tgt_seqs)

    def _pad(seqs: Tuple[List[int], ...], max_len: int) -> torch.Tensor:
        return torch.tensor(
            [s + [pad_id] * (max_len - len(s)) for s in seqs],
            dtype=torch.long,
        )

    return _pad(src_seqs, max_src), _pad(tgt_seqs, max_tgt)


def make_dataloader(
    split: DatasetSplit,
    tokenizer: GraphTokenizer,
    batch_size: int = _BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    dataset = GraphPathDataset(split, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def _make_masks(
    src: torch.Tensor,
    decoder_input: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = src.device
    tgt_len = decoder_input.size(1)

    src_mask = create_padding_mask(src, pad_id)                   # (B, 1, 1, src_len)
    tgt_pad_mask = create_padding_mask(decoder_input, pad_id)     # (B, 1, 1, tgt_len)
    causal_mask = create_causal_mask(tgt_len, device=device)      # (1, 1, tgt_len, tgt_len)
    tgt_self_attn_mask = causal_mask & tgt_pad_mask               # (B, 1, tgt_len, tgt_len)
    tgt_cross_attn_mask = src_mask                                # (B, 1, 1, src_len)

    return src_mask, tgt_self_attn_mask, tgt_cross_attn_mask


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: Transformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    scheduler: LambdaLR | None = None,
    *,
    grad_clip_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        decoder_input = tgt[:, :-1]   # (B, T-1)
        target_output = tgt[:, 1:]    # (B, T-1)

        src_mask, tgt_self_attn_mask, tgt_cross_attn_mask = _make_masks(
            src, decoder_input, criterion.ignore_index
        )

        optimizer.zero_grad()

        logits = model(
            src,
            decoder_input,
            src_mask=src_mask,
            tgt_self_attn_mask=tgt_self_attn_mask,
            tgt_cross_attn_mask=tgt_cross_attn_mask,
        )  # (B, T-1, vocab_size)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_output.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def _val_epoch(
    model: Transformer,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            decoder_input = tgt[:, :-1]
            target_output = tgt[:, 1:]

            src_mask, tgt_self_attn_mask, tgt_cross_attn_mask = _make_masks(
                src, decoder_input, criterion.ignore_index
            )

            logits = model(
                src,
                decoder_input,
                src_mask=src_mask,
                tgt_self_attn_mask=tgt_self_attn_mask,
                tgt_cross_attn_mask=tgt_cross_attn_mask,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_output.reshape(-1),
            )

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


# ---------------------------------------------------------------------------
# Top-level training function
# ---------------------------------------------------------------------------

def train_model(
    train_split: DatasetSplit,
    val_split: DatasetSplit,
    tokenizer: GraphTokenizer,
    *,
    n_epochs: int = 10,
    batch_size: int = _BATCH_SIZE,
    lr: float = _LR,
    weight_decay: float = _WEIGHT_DECAY,
    warmup_steps: int = _WARMUP_STEPS,
    n_layers: int = _N_LAYERS,
    d_model: int = _D_MODEL,
    n_heads: int = _N_HEADS,
    d_ff: int = _D_FF,
    dropout: float = _DROPOUT,
    grad_clip_norm: float = 1.0,
    device: torch.device | None = None,
    on_epoch_end: Optional[Callable[[int, nn.Module, Dict[str, List[float]]], None]] = None,
    lr_decay_start_epoch: int | None = None,
) -> Tuple[Transformer, Dict[str, List[float]]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    train_loader = make_dataloader(
        train_split, tokenizer, batch_size=batch_size, shuffle=True
    )

    # Build LR scheduler based on lr_decay_start_epoch
    if lr_decay_start_epoch is None:
        # Default: warmup only
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: _linear_warmup_schedule(step, warmup_steps),
        )
    else:
        # Warmup + decay
        steps_per_epoch = len(train_loader)
        decay_start_step = lr_decay_start_epoch * steps_per_epoch
        total_steps = n_epochs * steps_per_epoch

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: _warmup_and_decay_schedule(
                step, warmup_steps, decay_start_step, total_steps
            ),
        )
    val_loader = make_dataloader(
        val_split, tokenizer, batch_size=batch_size, shuffle=False
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss = _val_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:3d}/{n_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, model, history)

    return model, history
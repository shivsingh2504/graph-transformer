"""Tests for LR decay schedule in run 8."""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from data.dataset_generator import generate_dataset_split
from data.tokenizer import GraphTokenizer
from run_final_training import RunConfig, run
from train.train import _linear_warmup_schedule, _warmup_and_decay_schedule, train_model


def test_none_path_matches_old_schedule():
    """When lr_decay_start_epoch=None, schedule must match old warmup-only behavior."""
    optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=5e-4)
    from torch.optim.lr_scheduler import LambdaLR
    
    test_steps = [0, 3999, 4000, 1000000]
    
    for test_step in test_steps:
        optimizer = torch.optim.SGD([torch.tensor([1.0])], lr=5e-4)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: _linear_warmup_schedule(step, 4000))
        for _ in range(test_step):
            optimizer.step()
            scheduler.step()
        actual_lr = optimizer.param_groups[0]["lr"]
        expected_factor = _linear_warmup_schedule(test_step, 4000)
        expected_lr = 5e-4 * expected_factor
        assert abs(actual_lr - expected_lr) < 1e-9, f"Step {test_step}: {actual_lr} != {expected_lr}"


def test_decay_exact_values():
    """Verify exact schedule values at key points."""
    steps_per_epoch = 15625
    warmup_steps = 4000
    decay_start_epoch = 40
    n_epochs = 60
    decay_start_step = decay_start_epoch * steps_per_epoch
    total_steps = n_epochs * steps_per_epoch
    
    assert _warmup_and_decay_schedule(decay_start_step - 1, warmup_steps, decay_start_step, total_steps) == 1.0
    assert _warmup_and_decay_schedule(decay_start_step, warmup_steps, decay_start_step, total_steps) == 1.0
    
    midpoint = (decay_start_step + total_steps) // 2
    mid_factor = _warmup_and_decay_schedule(midpoint, warmup_steps, decay_start_step, total_steps)
    assert abs(mid_factor - 0.5) < 0.01
    
    last_factor = _warmup_and_decay_schedule(total_steps - 1, warmup_steps, decay_start_step, total_steps)
    expected_last = 1.0 / (total_steps - decay_start_step)
    assert abs(last_factor - expected_last) < 1e-6
    
    prev = 1.0
    for step in range(decay_start_step, total_steps, 1000):
        curr = _warmup_and_decay_schedule(step, warmup_steps, decay_start_step, total_steps)
        assert curr <= prev, f"Not monotone at step {step}: {curr} > {prev}"
        prev = curr


def test_end_to_end_lr_capture():
    """Train with decay and capture actual optimizer LR at each step."""
    train_split = generate_dataset_split(32, (5, 20), base_seed=0)
    val_split = generate_dataset_split(16, (5, 20), base_seed=1)
    tokenizer = GraphTokenizer()
    
    captured_lrs = []
    base_lr = 5e-4
    warmup_steps = 2
    lr_decay_start_epoch = 1
    n_epochs = 3
    batch_size = 16
    
    steps_per_epoch = len(train_split.examples) // batch_size
    decay_start_step = lr_decay_start_epoch * steps_per_epoch
    total_steps = n_epochs * steps_per_epoch
    
    from train.train import _train_epoch as original_train_epoch
    
    def capture_train_epoch(model, loader, optimizer, criterion, device, scheduler=None, *, grad_clip_norm=1.0):
        total_loss = 0.0
        n_batches = 0
        model.train()
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            decoder_input = tgt[:, :-1]
            target_output = tgt[:, 1:]
            from train.train import _make_masks
            src_mask, tgt_self_attn_mask, tgt_cross_attn_mask = _make_masks(
                src, decoder_input, criterion.ignore_index
            )
            optimizer.zero_grad()
            logits = model(src, decoder_input, src_mask=src_mask, 
                          tgt_self_attn_mask=tgt_self_attn_mask, 
                          tgt_cross_attn_mask=tgt_cross_attn_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            # Capture LR BEFORE scheduler advances (this is the LR used for this step)
            captured_lrs.append(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    with patch("train.train._train_epoch", capture_train_epoch):
        model, history = train_model(
            train_split, val_split, tokenizer,
            n_epochs=n_epochs, batch_size=batch_size, lr=base_lr,
            warmup_steps=warmup_steps, lr_decay_start_epoch=lr_decay_start_epoch
        )
    
    for step, captured_lr in enumerate(captured_lrs):
        expected_factor = _warmup_and_decay_schedule(step, warmup_steps, decay_start_step, total_steps)
        expected_lr = base_lr * expected_factor
        assert abs(captured_lr - expected_lr) < 1e-9, f"Step {step}: {captured_lr} != {expected_lr}"


def test_launcher_config_and_run_call():
    """Verify launcher asserts correct changes and calls run()."""
    cfg = RunConfig(n_epochs=60, lr_decay_start_epoch=40)
    baseline = RunConfig()
    
    import dataclasses
    changed = {
        f.name: (getattr(baseline, f.name), getattr(cfg, f.name))
        for f in dataclasses.fields(cfg)
        if getattr(baseline, f.name) != getattr(cfg, f.name)
    }
    assert changed == {"n_epochs": (30, 60), "lr_decay_start_epoch": (None, 40)}
    
    with patch("run_final_training.run") as mock_run:
        from run_training_run8 import main
        main()
        assert mock_run.called


def test_model_config_no_lr_decay():
    """Ensure lr_decay_start_epoch is NOT saved in model_config."""
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = RunConfig(num_train_examples=32, num_eval_examples=16, n_epochs=1, lr_decay_start_epoch=40)
        out_dir = Path(tmpdir) / "test_ckpt"
        
        result = run(cfg, out_dir)
        
        checkpoint = torch.load(out_dir / "final.pt", map_location="cpu")
        model_config = checkpoint["model_config"]
        train_config = checkpoint["train_config"]
        
        assert "lr_decay_start_epoch" not in model_config
        assert "lr_decay_start_epoch" in train_config
        assert train_config["lr_decay_start_epoch"] == 40

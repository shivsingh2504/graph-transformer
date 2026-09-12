
import torch
import torch.nn as nn
import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model.model import (
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
    Transformer,
)
from model.layers import create_padding_mask, create_causal_mask

VOCAB_SIZE = 50
PAD_ID = 0
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_LAYERS = 2
BATCH = 4
SRC_LEN = 10
TGT_LEN = 8

def test_encoder_layer_output_shape():
    layer = EncoderLayer(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    x = torch.randn(BATCH, SRC_LEN, D_MODEL)
    out = layer(x)
    assert out.shape == (BATCH, SRC_LEN, D_MODEL)


def test_encoder_layer_respects_padding_mask():
    layer = EncoderLayer(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    layer.eval()
    src_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    src_ids[:, -3:] = PAD_ID
    mask = create_padding_mask(src_ids, PAD_ID)
    x = torch.randn(BATCH, SRC_LEN, D_MODEL)
    out = layer(x, mask=mask)
    assert out.shape == (BATCH, SRC_LEN, D_MODEL)
    assert not torch.isnan(out).any()


def test_encoder_layer_deterministic_at_zero_dropout():
    layer = EncoderLayer(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    layer.eval()
    x = torch.randn(BATCH, SRC_LEN, D_MODEL)
    out1 = layer(x)
    out2 = layer(x)
    assert torch.allclose(out1, out2)

def test_decoder_layer_output_shape():
    layer = DecoderLayer(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    x = torch.randn(BATCH, TGT_LEN, D_MODEL)
    enc_out = torch.randn(BATCH, SRC_LEN, D_MODEL)
    out = layer(x, enc_out)
    assert out.shape == (BATCH, TGT_LEN, D_MODEL)


def test_decoder_layer_with_causal_and_padding_masks():
    layer = DecoderLayer(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    layer.eval()
    x = torch.randn(BATCH, TGT_LEN, D_MODEL)
    enc_out = torch.randn(BATCH, SRC_LEN, D_MODEL)

    causal = create_causal_mask(TGT_LEN)
    tgt_ids = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN))
    pad = create_padding_mask(tgt_ids, PAD_ID)
    self_attn_mask = causal & pad

    src_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    cross_attn_mask = create_padding_mask(src_ids, PAD_ID)

    out = layer(x, enc_out, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask)
    assert out.shape == (BATCH, TGT_LEN, D_MODEL)
    assert not torch.isnan(out).any()


def test_decoder_layer_causal_mask_blocks_future():
    layer = DecoderLayer(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    layer.eval()
    enc_out = torch.randn(1, SRC_LEN, D_MODEL)
    causal = create_causal_mask(TGT_LEN)

    x1 = torch.randn(1, TGT_LEN, D_MODEL)
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(D_MODEL)  

    out1 = layer(x1, enc_out, self_attn_mask=causal)
    out2 = layer(x2, enc_out, self_attn_mask=causal)


    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)

def test_encoder_n_layers_stacked():
    enc = Encoder(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                   n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    assert len(enc.layers) == N_LAYERS
    src_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    out = enc(src_ids)
    assert out.shape == (BATCH, SRC_LEN, D_MODEL)


def test_decoder_n_layers_stacked():
    dec = Decoder(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                   n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    assert len(dec.layers) == N_LAYERS
    tgt_ids = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN))
    enc_out = torch.randn(BATCH, SRC_LEN, D_MODEL)
    out = dec(tgt_ids, enc_out)
    assert out.shape == (BATCH, TGT_LEN, D_MODEL)


def test_encoder_layers_are_independent_instances():
    enc = Encoder(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                   n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    w0 = enc.layers[0].self_attn.W_q.weight
    w1 = enc.layers[1].self_attn.W_q.weight
    assert w0.data_ptr() != w1.data_ptr()
    assert not torch.equal(w0, w1)

def test_transformer_output_shape_is_logits_over_vocab():
    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                         n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    src_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    tgt_ids = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN))
    logits = model(src_ids, tgt_ids)
    assert logits.shape == (BATCH, TGT_LEN, VOCAB_SIZE)


def test_transformer_output_is_raw_logits_not_probabilities():
    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                         n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    src_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    tgt_ids = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN))
    logits = model(src_ids, tgt_ids)
    row_sums = logits.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert (logits < 0).any() or (logits > 1).any()


def test_transformer_encode_decode_match_forward():
    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                         n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    model.eval()
    src_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    tgt_ids = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN))

    logits_forward = model(src_ids, tgt_ids)

    enc_out = model.encode(src_ids)
    logits_decode = model.decode(tgt_ids, enc_out)

    assert torch.allclose(logits_forward, logits_decode, atol=1e-6)


def test_transformer_full_masking_pipeline():
    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                         n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    model.eval()

    src_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    src_ids[:, -2:] = PAD_ID
    tgt_ids = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN))
    tgt_ids[:, -1:] = PAD_ID

    src_mask = create_padding_mask(src_ids, PAD_ID)
    tgt_pad_mask = create_padding_mask(tgt_ids, PAD_ID)
    causal = create_causal_mask(TGT_LEN)
    tgt_self_attn_mask = causal & tgt_pad_mask
    tgt_cross_attn_mask = src_mask

    logits = model(
        src_ids, tgt_ids,
        src_mask=src_mask,
        tgt_self_attn_mask=tgt_self_attn_mask,
        tgt_cross_attn_mask=tgt_cross_attn_mask,
    )
    assert logits.shape == (BATCH, TGT_LEN, VOCAB_SIZE)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_transformer_gradients_flow_to_embeddings():
    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                         n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    src_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    tgt_ids = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN))

    logits = model(src_ids, tgt_ids)
    loss = logits.sum()
    loss.backward()

    assert model.encoder.embedding.embedding.weight.grad is not None
    assert not torch.all(model.encoder.embedding.embedding.weight.grad == 0)


def test_transformer_deterministic_at_zero_dropout():
    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                         n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    model.eval()
    src_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    tgt_ids = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN))

    out1 = model(src_ids, tgt_ids)
    out2 = model(src_ids, tgt_ids)
    assert torch.allclose(out1, out2)


def test_transformer_nonzero_dropout_is_stochastic_in_train_mode():
    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                         n_heads=N_HEADS, d_ff=D_FF, dropout=0.5)
    model.train()
    src_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    tgt_ids = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN))

    torch.manual_seed(0)
    out1 = model(src_ids, tgt_ids)
    torch.manual_seed(1)
    out2 = model(src_ids, tgt_ids)

    assert not torch.allclose(out1, out2)


def test_transformer_variable_sequence_lengths():
    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, d_model=D_MODEL,
                         n_heads=N_HEADS, d_ff=D_FF, dropout=0.0)
    for src_len, tgt_len in [(5, 4), (20, 15), (50, 40)]:
        src_ids = torch.randint(1, VOCAB_SIZE, (2, src_len))
        tgt_ids = torch.randint(1, VOCAB_SIZE, (2, tgt_len))
        logits = model(src_ids, tgt_ids)
        assert logits.shape == (2, tgt_len, VOCAB_SIZE)
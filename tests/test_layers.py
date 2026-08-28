import math
import pytest
import torch

from model.layers import (
    PositionalEncoding,
    MultiHeadAttention,
    PositionwiseFeedForward,
    TokenEmbedding,
    create_padding_mask,
    create_causal_mask,
)

D_MODEL = 128
N_HEADS = 4
D_FF = 512
BATCH = 2
SEQ_LEN = 10
VOCAB_SIZE = 65
PAD_ID = 0


class TestCreatePaddingMask:

    def test_output_shape(self) -> None:
        ids = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
        mask = create_padding_mask(ids, PAD_ID)
        assert mask.shape == (BATCH, 1, 1, SEQ_LEN)

    def test_dtype_is_bool(self) -> None:
        ids = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
        mask = create_padding_mask(ids, PAD_ID)
        assert mask.dtype == torch.bool

    def test_real_tokens_are_true(self) -> None:
        ids = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
        mask = create_padding_mask(ids, PAD_ID)
        assert mask.all()

    def test_padding_tokens_are_false(self) -> None:
        ids = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long)
        mask = create_padding_mask(ids, PAD_ID)
        assert not mask.any()

    def test_mixed_batch(self) -> None:
        ids = torch.ones(2, 5, dtype=torch.long)
        ids[0, 2:] = PAD_ID
        mask = create_padding_mask(ids, PAD_ID)
        assert mask[0, 0, 0, :2].all()
        assert not mask[0, 0, 0, 2:].any()
        assert mask[1, 0, 0, :].all()

    def test_hand_constructed_example(self) -> None:
        ids = torch.tensor([[1, 2, PAD_ID, PAD_ID]])
        mask = create_padding_mask(ids, PAD_ID)
        expected = torch.tensor([[[[True, True, False, False]]]])
        assert torch.equal(mask, expected)


class TestCreateCausalMask:

    def test_output_shape(self) -> None:
        mask = create_causal_mask(SEQ_LEN)
        assert mask.shape == (1, 1, SEQ_LEN, SEQ_LEN)

    def test_dtype_is_bool(self) -> None:
        mask = create_causal_mask(SEQ_LEN)
        assert mask.dtype == torch.bool

    def test_lower_triangular_structure(self) -> None:
        n = 5
        mask = create_causal_mask(n)
        for i in range(n):
            for j in range(n):
                expected = (j <= i)
                assert mask[0, 0, i, j].item() == expected

    def test_diagonal_is_true(self) -> None:
        mask = create_causal_mask(6)
        for i in range(6):
            assert mask[0, 0, i, i].item() is True

    def test_above_diagonal_is_false(self) -> None:
        mask = create_causal_mask(6)
        for i in range(6):
            for j in range(i + 1, 6):
                assert mask[0, 0, i, j].item() is False

    def test_hand_constructed_size_3(self) -> None:
        mask = create_causal_mask(3)
        expected = torch.tensor([[[[
            True,  False, False,
            True,  True,  False,
            True,  True,  True,
        ]]]], dtype=torch.bool).view(1, 1, 3, 3)
        assert torch.equal(mask, expected)

    def test_device_argument_respected(self) -> None:
        cpu_mask = create_causal_mask(4, device=torch.device("cpu"))
        assert cpu_mask.device.type == "cpu"


class TestPositionalEncoding:

    @pytest.fixture
    def pe(self) -> PositionalEncoding:
        return PositionalEncoding(d_model=D_MODEL, dropout=0.0)

    def test_output_shape(self, pe: PositionalEncoding) -> None:
        x = torch.zeros(BATCH, SEQ_LEN, D_MODEL)
        out = pe(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_deterministic_same_input_same_output(self, pe: PositionalEncoding) -> None:
        pe.eval()
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        assert torch.equal(pe(x), pe(x))

    def test_pe_is_not_noop(self, pe: PositionalEncoding) -> None:
        pe.eval()
        x = torch.zeros(1, SEQ_LEN, D_MODEL)
        out = pe(x)
        assert not torch.all(out == 0)

    def test_ood_length_200(self, pe: PositionalEncoding) -> None:
        pe.eval()
        x = torch.zeros(1, 200, D_MODEL)
        out = pe(x)
        assert out.shape == (1, 200, D_MODEL)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_ood_length_1000(self, pe: PositionalEncoding) -> None:
        pe.eval()
        x = torch.zeros(1, 1000, D_MODEL)
        out = pe(x)
        assert out.shape == (1, 1000, D_MODEL)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_sinusoidal_values_at_position_0(self, pe: PositionalEncoding) -> None:
        pe.eval()
        x = torch.zeros(1, 1, D_MODEL)
        out = pe(x)
        pe_at_0 = out[0, 0]
        assert torch.allclose(pe_at_0[0::2], torch.zeros(D_MODEL // 2), atol=1e-6)
        assert torch.allclose(pe_at_0[1::2], torch.ones(D_MODEL // 2), atol=1e-6)

    def test_sinusoidal_value_at_position_1_dim_0(self, pe: PositionalEncoding) -> None:
        pe.eval()
        x = torch.zeros(1, 5, D_MODEL)
        out = pe(x)
        expected = math.sin(1.0)
        assert torch.isclose(out[0, 1, 0], torch.tensor(expected), atol=1e-5)

    def test_different_positions_have_distinct_encodings(self, pe: PositionalEncoding) -> None:
        pe.eval()
        n = 20
        x = torch.zeros(1, n, D_MODEL)
        out = pe(x)
        for i in range(n):
            for j in range(i + 1, n):
                assert not torch.allclose(out[0, i], out[0, j])

    def test_pe_is_added_to_input_not_replaced(self, pe: PositionalEncoding) -> None:
        pe.eval()
        x = torch.randn(1, SEQ_LEN, D_MODEL)
        out = pe(x)
        pe_values = pe.pe[:, :SEQ_LEN, :]
        assert torch.allclose(out, x + pe_values, atol=1e-6)


class TestMultiHeadAttention:

    @pytest.fixture
    def mha(self) -> MultiHeadAttention:
        return MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0)

    def test_self_attention_output_shape(self, mha: MultiHeadAttention) -> None:
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = mha(x, x, x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_cross_attention_output_shape(self, mha: MultiHeadAttention) -> None:
        seq_q, seq_k = 7, 15
        query = torch.randn(BATCH, seq_q, D_MODEL)
        context = torch.randn(BATCH, seq_k, D_MODEL)
        out = mha(query, context, context)
        assert out.shape == (BATCH, seq_q, D_MODEL)

    def test_output_shape_with_mask(self, mha: MultiHeadAttention) -> None:
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        ids = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
        ids[0, -2:] = PAD_ID
        mask = create_padding_mask(ids, PAD_ID)
        out = mha(x, x, x, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_return_attn_weights_shape(self, mha: MultiHeadAttention) -> None:
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out, weights = mha(x, x, x, return_attn_weights=True)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert weights.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_masked_positions_receive_zero_attention_weight(
        self, mha: MultiHeadAttention
    ) -> None:
        mha.eval()
        seq_q, seq_k = 1, 5
        query = torch.randn(1, seq_q, D_MODEL)
        key = torch.randn(1, seq_k, D_MODEL)
        value = torch.randn(1, seq_k, D_MODEL)

        mask = torch.zeros(1, 1, seq_q, seq_k, dtype=torch.bool)
        mask[:, :, :, 0] = True

        _, weights = mha(query, key, value, mask=mask, return_attn_weights=True)

        assert torch.allclose(
            weights[:, :, :, 1:],
            torch.zeros_like(weights[:, :, :, 1:]),
            atol=1e-5,
        )
        assert torch.allclose(
            weights[:, :, :, 0],
            torch.ones_like(weights[:, :, :, 0]),
            atol=1e-5,
        )

    def test_all_positions_masked_except_one_output_equals_value(
        self, mha: MultiHeadAttention
    ) -> None:
        mha.eval()
        mask = torch.zeros(1, 1, 1, 4, dtype=torch.bool)
        mask[:, :, :, 2] = True
        query = torch.randn(1, 1, D_MODEL)
        keys = torch.randn(1, 4, D_MODEL)
        out = mha(query, keys, keys, mask=mask)
        assert out.shape == (1, 1, D_MODEL)
        assert not torch.any(torch.isnan(out))

    def test_causal_mask_blocks_future_positions(
        self, mha: MultiHeadAttention
    ) -> None:
        mha.eval()
        seq_len = 4
        x = torch.randn(1, seq_len, D_MODEL)
        x_modified = x.clone()
        x_modified[:, 1:, :] = torch.randn(1, seq_len - 1, D_MODEL)

        causal = create_causal_mask(seq_len)

        out_orig, _ = mha(x, x, x, mask=causal, return_attn_weights=True)
        out_modified, _ = mha(
            x_modified, x_modified, x_modified, mask=causal, return_attn_weights=True
        )
        assert torch.allclose(out_orig[:, 0, :], out_modified[:, 0, :], atol=1e-5)

        out_no_mask_orig = mha(x, x, x)
        out_no_mask_mod = mha(x_modified, x_modified, x_modified)
        assert not torch.allclose(
            out_no_mask_orig[:, 0, :], out_no_mask_mod[:, 0, :], atol=1e-5
        )

    def test_causal_mask_allows_attending_to_all_past(
        self, mha: MultiHeadAttention
    ) -> None:
        mha.eval()
        seq_len = 4
        x = torch.randn(1, seq_len, D_MODEL)
        x_modified = x.clone()
        x_modified[:, 0, :] = torch.randn(D_MODEL)

        causal = create_causal_mask(seq_len)
        out_orig = mha(x, x, x, mask=causal)
        out_mod = mha(x_modified, x_modified, x_modified, mask=causal)

        assert not torch.allclose(out_orig[:, 3, :], out_mod[:, 3, :], atol=1e-5)

    def test_padding_mask_applied_correctly(self, mha: MultiHeadAttention) -> None:
        mha.eval()
        ids = torch.tensor([[1, 2, PAD_ID, PAD_ID]])
        mask = create_padding_mask(ids, PAD_ID)

        x_a = torch.randn(1, 4, D_MODEL)
        x_b = x_a.clone()
        x_b[:, 2:, :] = torch.randn(1, 2, D_MODEL)

        out_a = mha(x_a, x_a, x_a, mask=mask)
        out_b = mha(x_b, x_b, x_b, mask=mask)

        assert torch.allclose(out_a[:, :2, :], out_b[:, :2, :], atol=1e-5)


class TestPositionwiseFeedForward:

    @pytest.fixture
    def ffn(self) -> PositionwiseFeedForward:
        return PositionwiseFeedForward(d_model=D_MODEL, d_ff=D_FF, dropout=0.0)

    def test_output_shape(self, ffn: PositionwiseFeedForward) -> None:
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        assert ffn(x).shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_output_shape_seq1(self, ffn: PositionwiseFeedForward) -> None:
        x = torch.randn(BATCH, 1, D_MODEL)
        assert ffn(x).shape == (BATCH, 1, D_MODEL)

    def test_is_not_noop(self, ffn: PositionwiseFeedForward) -> None:
        ffn.eval()
        x = torch.randn(1, SEQ_LEN, D_MODEL)
        assert not torch.allclose(ffn(x), x)

    def test_parameter_count(self) -> None:
        ffn = PositionwiseFeedForward(d_model=D_MODEL, d_ff=D_FF)
        param_shapes = sorted([p.shape for p in ffn.parameters()])
        expected = sorted([
            torch.Size([D_FF, D_MODEL]),
            torch.Size([D_FF]),
            torch.Size([D_MODEL, D_FF]),
            torch.Size([D_MODEL]),
        ])
        assert param_shapes == expected

    def test_processes_each_position_independently(self) -> None:
        ffn = PositionwiseFeedForward(d_model=D_MODEL, d_ff=D_FF, dropout=0.0)
        ffn.eval()
        x = torch.randn(1, 5, D_MODEL)
        x_mod = x.clone()
        x_mod[:, 2, :] = torch.randn(D_MODEL)

        out = ffn(x)
        out_mod = ffn(x_mod)

        for pos in [0, 1, 3, 4]:
            assert torch.allclose(out[:, pos, :], out_mod[:, pos, :], atol=1e-6)
        assert not torch.allclose(out[:, 2, :], out_mod[:, 2, :])


class TestTokenEmbedding:

    @pytest.fixture
    def embed(self) -> TokenEmbedding:
        return TokenEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL, dropout=0.0)

    def test_output_shape(self, embed: TokenEmbedding) -> None:
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = embed(ids)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_vocab_size_parameter_is_respected(self) -> None:
        e1 = TokenEmbedding(vocab_size=65, d_model=D_MODEL)
        e2 = TokenEmbedding(vocab_size=100, d_model=D_MODEL)
        assert e1.embedding.weight.shape == (65, D_MODEL)
        assert e2.embedding.weight.shape == (100, D_MODEL)

    def test_different_vocab_sizes_produce_correct_shapes(self) -> None:
        for vs in [10, 65, 256]:
            em = TokenEmbedding(vocab_size=vs, d_model=D_MODEL, dropout=0.0)
            ids = torch.randint(0, vs, (1, SEQ_LEN))
            assert em(ids).shape == (1, SEQ_LEN, D_MODEL)

    def test_embedding_is_scaled_by_sqrt_d_model(self) -> None:
        em = TokenEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL, dropout=0.0)
        em.eval()
        raw = em.embedding(torch.tensor([[1]]))
        out = em(torch.tensor([[1]]))

        pe_at_0 = em.pos_encoding.pe[:, :1, :]
        expected = raw * math.sqrt(D_MODEL) + pe_at_0
        assert torch.allclose(out, expected, atol=1e-5)

    def test_deterministic_in_eval_mode(self, embed: TokenEmbedding) -> None:
        embed.eval()
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        assert torch.equal(embed(ids), embed(ids))

    def test_output_is_finite(self, embed: TokenEmbedding) -> None:
        embed.eval()
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = embed(ids)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_pad_token_embeds_correctly(self, embed: TokenEmbedding) -> None:
        embed.eval()
        ids = torch.zeros(1, 5, dtype=torch.long)
        out = embed(ids)
        assert out.shape == (1, 5, D_MODEL)
        assert not torch.any(torch.isnan(out))
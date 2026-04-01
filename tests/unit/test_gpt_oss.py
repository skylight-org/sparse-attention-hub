import torch
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import SinkMasker, SinkMaskerConfig
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import AttentionTensorDimensions


def test_gpt_oss_sink_mask():
    batch_size = 1
    num_heads = 1
    seq_len = 8

    keys = torch.randn(batch_size, num_heads, seq_len, 64)
    queries = torch.randn(batch_size, num_heads, seq_len, 64)
    values = torch.randn(batch_size, num_heads, seq_len, 64)

    tensor_dims = AttentionTensorDimensions(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_queries=seq_len,
        seq_len_keys=seq_len
    )

    shape = (batch_size, num_heads, seq_len, seq_len)
    dense_mask = torch.zeros(shape)

    previous_mask = Mask.create_mask_from_dense_mask(
        shape,
        dense_mask,
        torch.float32
    )

    config = SinkMaskerConfig(sink_size=2, is_gpt_oss=True)
    masker = SinkMasker(config)

    output_mask = masker.add_mask(
        keys,
        queries,
        values,
        attention_mask=None,
        scaling=1.0,
        dropout=0.0,
        sparse_meta_data={},
        previous_mask=previous_mask
    )

    dense = output_mask.get_dense_mask()
    # Check: sink tokens respect causal masking (lower triangular)
    for i in range(seq_len):
        for j in range(2):  # sink tokens
            if j <= i:
                assert dense[..., i, j] == 1
            else:
                assert dense[..., i, j] == 0

    # Check: rest are not all ones
    assert not torch.all(dense == 1)

    # Check: causal masking (no future attention)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert dense[..., i, j] == 0
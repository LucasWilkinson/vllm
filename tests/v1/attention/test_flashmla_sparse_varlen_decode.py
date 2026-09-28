# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashMLA sparse decode under adaptive verification (ragged decode rows).

Adaptive verification trims each request's drafts on device, so FULL decode
graphs replay batches whose per-request query lengths vary, and the host query
lengths only carry the right total. FlashMLA sparse handles this by running the
decode rows as one flat batch of single-token rows (every row carries its own
top-k) and, under HiSparse, by mapping rows to requests from the device
query_start_loc.

Everything here except the last test runs without a GPU.
"""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import flashmla_sparse
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.worker.gpu.attn_utils import get_varlen_cudagraph_unsupported_backend

NUM_SPEC = 7
DECODE_WIDTH = NUM_SPEC + 1


def _config(adaptive: bool | None, dcp: int = 1, pcp: int = 1):
    speculative_config = None
    if adaptive is not None:
        speculative_config = SimpleNamespace(
            enable_adaptive_verification=adaptive,
            num_speculative_tokens=NUM_SPEC,
            parallel_drafting=False,
        )
    return SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        use_v2_model_runner=True,
    )


@pytest.mark.parametrize(
    "adaptive,dcp,pcp,expected",
    [
        (None, 1, 1, None),
        (False, 1, 1, None),
        (True, 1, 1, DECODE_WIDTH),
        (True, 2, 1, None),
        (True, 1, 2, None),
    ],
)
def test_varlen_cudagraph_bound(adaptive, dcp, pcp, expected):
    """Only the flattened decode layout replays ragged verification; the
    uniform support level itself does not change."""
    config = _config(adaptive, dcp, pcp)
    spec = object()
    builder_cls = FlashMLASparseMetadataBuilder
    assert (
        builder_cls.get_cudagraph_support(config, spec)
        == AttentionCGSupport.UNIFORM_BATCH
    )
    assert builder_cls.get_varlen_cudagraph_max_query_len(config, spec) == expected
    assert flashmla_sparse.flashmla_sparse_flattens_decodes(config) == (
        expected is not None
    )


def test_adaptive_verification_accepts_flashmla_sparse():
    group = SimpleNamespace(
        layer_names=["model.layers.0.self_attn.attn"],
        backend=FlashMLASparseBackend,
        kv_cache_spec=object(),
        get_metadata_builder=lambda _idx=0: FlashMLASparseMetadataBuilder,
    )
    assert (
        get_varlen_cudagraph_unsupported_backend([[group]], _config(True), DECODE_WIDTH)
        is None
    )
    assert get_varlen_cudagraph_unsupported_backend(
        [[group]], _config(False), DECODE_WIDTH
    ) == (FlashMLASparseBackend.__name__, None)


def _common(query_lens: list[int], max_query_len: int):
    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(query_lens, dtype=torch.int32), 0)
    return SimpleNamespace(
        max_query_len=max_query_len,
        num_reqs=len(query_lens),
        num_actual_tokens=int(qsl[-1]),
        query_start_loc_cpu=qsl,
    )


@pytest.mark.parametrize(
    "query_lens",
    [
        # Varlen capture dummy past max_num_reqs: remainder spread over the tail.
        [1, 1, 2, 2],
        # Host view of an adaptive batch: draft budget spread evenly.
        [4, 4, 3, 3],
    ],
)
def test_ragged_decodes_stay_decodes_without_uniform_split(query_lens):
    """A uniform split would turn ragged decode rows into prefills, baking a
    prefill route into the decode graph; the flattened layout keeps them."""
    common = _common(query_lens, DECODE_WIDTH)
    num_tokens = sum(query_lens)
    assert split_decodes_and_prefills(
        common, decode_threshold=DECODE_WIDTH, require_uniform=False
    ) == (len(query_lens), 0, num_tokens, 0)
    num_decodes, *_ = split_decodes_and_prefills(
        common, decode_threshold=DECODE_WIDTH, require_uniform=True
    )
    assert num_decodes < len(query_lens)


def _fake_swap_in(request_ids, *, logical_topk_indices, **kwargs):
    """Physical slot = logical + 1000 * (request + 1); -1 stays -1."""
    rows = request_ids.to(torch.int32).unsqueeze(1) + 1
    physical = torch.where(
        logical_topk_indices >= 0,
        logical_topk_indices + 1000 * rows,
        torch.full_like(logical_topk_indices, -1),
    )
    counts = (logical_topk_indices >= 0).sum(dim=1).to(torch.int32)
    if kwargs.get("attention_indices_out") is not None:
        kwargs["attention_indices_out"].copy_(physical)
    if kwargs.get("valid_counts_out") is not None:
        kwargs["valid_counts_out"].copy_(counts)
    if kwargs.get("return_valid_counts"):
        return physical, counts
    return physical


def _index_group(max_rows: int, topk: int, num_layers: int = 1):
    group = object.__new__(HiSparseMLAIndexGroup)
    group.caches = [
        SimpleNamespace(
            source_block_table=torch.zeros((max_rows, 1), dtype=torch.int32),
            swap_in=MagicMock(side_effect=_fake_swap_in),
        )
        for _ in range(num_layers)
    ]
    # Garbage, as torch.empty would leave it on the first replay.
    group.physical_topk_indices = torch.full((max_rows + 1, topk), 123456789)
    group.physical_topk_indices = group.physical_topk_indices.to(torch.int32)
    group.valid_topk_counts = torch.full((max_rows + 1,), 77, dtype=torch.int32)
    group.request_ids = torch.arange(max_rows, dtype=torch.int32)
    return group


def _reference(logical: torch.Tensor, device_lens: list[int]):
    physical = torch.full_like(logical, -1)
    counts = torch.zeros(logical.shape[0], dtype=torch.int32)
    start = 0
    for req, n in enumerate(device_lens):
        rows = logical[start : start + n]
        physical[start : start + n] = torch.where(
            rows >= 0, rows + 1000 * (req + 1), torch.full_like(rows, -1)
        )
        counts[start : start + n] = (rows >= 0).sum(dim=1)
        start += n
    return physical, counts


@pytest.mark.parametrize(
    "device_lens,num_padded_tokens",
    [
        # Ragged, with a padding request and trailing token padding.
        ([3, 1, 4, 0], 10),
        # Looks uniform to the host (2 requests x 4 rows = 8 tokens) but the
        # device trimmed request 0 to one row; 3 rows are token padding.
        ([1, 4], 8),
        # Genuinely uniform: the varlen path must agree with the uniform one.
        ([4, 4], 8),
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_hisparse_varlen_conversion_follows_device_rows(
    device_lens, num_padded_tokens, seed
):
    torch.manual_seed(seed)
    topk = 6
    num_decodes = len(device_lens)
    decode_width = 4
    logical = torch.randint(-1, 50, (num_padded_tokens, topk), dtype=torch.int32)
    qsl = torch.zeros(num_decodes + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(device_lens, dtype=torch.int32), 0)
    metadata = SimpleNamespace(
        num_decodes=num_decodes,
        decode_max_query_len=decode_width,
        decode_varlen=True,
        query_start_loc=qsl,
        block_size=64,
    )
    group = _index_group(max_rows=16, topk=topk, num_layers=2)

    physical, counts = group.convert_decode_logical_to_physical_topk(
        0, logical, metadata, return_valid_counts=True
    )
    expected_physical, expected_counts = _reference(logical, device_lens)
    torch.testing.assert_close(physical, expected_physical)
    torch.testing.assert_close(counts, expected_counts)
    # One swap-in per step bound, over every request row.
    assert group.caches[0].swap_in.call_count == decode_width

    # Followers reuse the leader's conversion.
    follower = group.convert_decode_logical_to_physical_topk(
        1, logical, metadata, return_valid_counts=False
    )
    torch.testing.assert_close(follower, expected_physical)


def test_hisparse_uniform_shape_needs_varlen_flag():
    """Without the flag a uniform-looking shape takes the reshape path, which
    assumes request r owns rows [r*q, (r+1)*q); that is what a graph captured
    for adaptive verification must not bake in."""
    topk = 3
    device_lens = [1, 4]
    logical = torch.arange(8 * topk, dtype=torch.int32).view(8, topk)
    qsl = torch.tensor([0, 1, 5], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=2,
        decode_max_query_len=4,
        query_start_loc=qsl,
        block_size=64,
    )
    expected, _ = _reference(logical, device_lens)

    uniform_group = _index_group(max_rows=8, topk=topk)
    uniform = uniform_group.convert_decode_logical_to_physical_topk(
        0, logical, metadata, return_valid_counts=False
    )
    assert not torch.equal(uniform, expected)

    varlen_group = _index_group(max_rows=8, topk=topk)
    varlen = varlen_group.convert_decode_logical_to_physical_topk(
        0, logical, metadata, return_valid_counts=False, varlen=True
    )
    torch.testing.assert_close(varlen, expected)


def _capture_kernel(shapes):
    def run_kernel(self, q, kv_c_and_k_pe_cache, topk_indices, kernel_metadata):
        shapes.append((tuple(q.shape), tuple(topk_indices.shape)))
        # Stand-in output that keeps row identity: (B, S, H, 1).
        return q[..., :1] + topk_indices[..., :1, None].to(q.dtype), None

    return run_kernel


def test_host_backed_decode_flattens_rows():
    num_tokens, num_heads, head_dim, topk = 5, 2, 4, 3
    device_lens = [3, 1, 1]
    q = torch.randn(num_tokens, num_heads, head_dim)
    logical = torch.randint(0, 20, (num_tokens, topk), dtype=torch.int32)
    qsl = torch.tensor([0, 3, 4, 5], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=3,
        decode_max_query_len=DECODE_WIDTH,
        decode_varlen=True,
        query_start_loc=qsl,
        block_size=64,
    )
    group = _index_group(max_rows=32, topk=topk)
    group.physical_kv_cache = lambda _layer: torch.empty(0)
    shapes: list = []
    impl = SimpleNamespace(index_group=group, index_group_index=0)
    impl._fp8_flash_mla_kernel = MethodType(_capture_kernel(shapes), impl)

    out = FlashMLASparseImpl._host_backed_fp8_decode(
        impl, q, logical, metadata, SimpleNamespace(), 3, DECODE_WIDTH, flatten=True
    )

    assert shapes == [((1, num_tokens, num_heads, head_dim), (1, num_tokens, topk))]
    expected_physical, _ = _reference(logical, device_lens)
    torch.testing.assert_close(
        out, q[..., :1] + expected_physical[:, :1, None].to(q.dtype)
    )


def test_separate_path_flat_decode_uses_one_flat_batch():
    num_tokens, num_heads, head_dim, topk = 7, 2, 4, 3
    q = torch.randn(num_tokens, num_heads, head_dim)
    topk_indices = torch.randint(0, 20, (num_tokens, topk), dtype=torch.int32)
    kernel_metadata = object()
    fp8_metadata = FlashMLASparseMetadata.FP8SeparatePrefillDecode(
        num_decodes=3,
        num_decode_tokens=num_tokens,
        decode=FlashMLASparseMetadata.FP8SeparatePrefillDecode.Decode(
            seq_lens=torch.zeros(3, dtype=torch.int32),
            kernel_metadata=kernel_metadata,
            decode_query_len=DECODE_WIDTH,
            flatten=True,
        ),
    )
    shapes: list = []
    impl = SimpleNamespace(index_group=None, pcp_dcp_kv_gather=False)
    impl._fp8_flash_mla_kernel = MethodType(_capture_kernel(shapes), impl)
    impl._convert_logical_to_physical_topk = MethodType(
        lambda self, topk, _meta, **_kw: (topk + 100, None), impl
    )
    attn_metadata = SimpleNamespace(fp8_extra_metadata=fp8_metadata)

    out, lse = FlashMLASparseImpl._forward_fp8_kv_separate_prefill_decode(
        impl, q, torch.empty(0), topk_indices, attn_metadata
    )

    assert lse is None
    assert shapes == [((1, num_tokens, num_heads, head_dim), (1, num_tokens, topk))]
    torch.testing.assert_close(
        out, q[..., :1] + (topk_indices + 100)[:, :1, None].to(q.dtype)
    )


def test_builder_flat_decode_metadata_ignores_host_split(monkeypatch):
    """The host lengths of an adaptive batch are ragged and not the device ones;
    the flattened build must neither assert on them nor size by them."""
    sched_meta = object()
    monkeypatch.setattr(flashmla_sparse, "get_mla_metadata", lambda: (sched_meta, None))
    builder = object.__new__(FlashMLASparseMetadataBuilder)
    builder.flatten_decodes = True
    builder.pcp_dcp_kv_gather = False
    builder.max_model_len_tensor = torch.full((16,), 4096, dtype=torch.int32)
    builder.dummy_block_table = torch.zeros((16, 1), dtype=torch.int32)
    query_lens = [4, 4, 3, 3]
    qsl = torch.tensor([0, 4, 8, 11, 14], dtype=torch.int32)
    common = SimpleNamespace(
        num_actual_tokens=14,
        query_start_loc_cpu=qsl,
        seq_lens=torch.full((len(query_lens),), 100, dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        num_decodes=4,
        num_prefills=0,
        num_decode_tokens=14,
        decode_max_query_len=DECODE_WIDTH,
    )

    fp8 = builder._build_fp8_separate_prefill_decode(common, metadata)

    assert fp8.num_decodes == 4
    assert fp8.decode is not None
    assert fp8.decode.flatten
    assert fp8.decode.decode_query_len == DECODE_WIDTH
    assert fp8.decode.kernel_metadata.scheduler_metadata is sched_meta
    # The flat call is a single batch entry.
    assert fp8.decode.kernel_metadata.cache_lens.shape == (1,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("batch_size,query_len", [(4, 8), (13, 3)])
def test_fp8_sparse_decode_flat_matches_per_request(batch_size, query_len):
    """The FP8 sparse decode kernel attends only each row's own top-k, so the
    (1, B*S) flat call must match the (B, S) spec-decode layout it replaces."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache,
        get_mla_metadata,
        is_flashmla_sparse_supported,
    )

    supported, reason = is_flashmla_sparse_supported()
    if not supported:
        pytest.skip(reason)
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, kv_lora_rank, rope_dim, topk, block_size = 64, 512, 64, 128, 64
    num_blocks = 32
    num_slots = num_blocks * block_size
    kv_c = torch.randn(num_slots, kv_lora_rank, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(num_slots, rope_dim, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(num_blocks, block_size, 656, dtype=torch.uint8, device=device)
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        cache,
        torch.arange(num_slots, device=device),
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1, device=device),
    )
    num_tokens = batch_size * query_len
    q = torch.randn(
        num_tokens,
        num_heads,
        kv_lora_rank + rope_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    indices = torch.randint(0, num_slots, (num_tokens, topk), device=device)
    indices = indices.to(torch.int32)
    indices[::5, topk // 2 :] = -1  # some short rows
    k_cache = cache.view(torch.uint8).unsqueeze(-2)

    def run(b: int, s: int) -> torch.Tensor:
        sched_meta, _ = get_mla_metadata()
        out, _ = flash_mla_with_kvcache(
            q=q.view(b, s, num_heads, -1),
            k_cache=k_cache,
            block_table=torch.zeros((b, 1), dtype=torch.int32, device=device),
            head_dim_v=kv_lora_rank,
            cache_seqlens=torch.full((b,), num_slots, dtype=torch.int32, device=device),
            tile_scheduler_metadata=sched_meta,
            is_fp8_kvcache=True,
            indices=indices.view(b, s, topk),
            softmax_scale=0.1,
        )
        return out.reshape(num_tokens, num_heads, kv_lora_rank)

    torch.testing.assert_close(run(1, num_tokens), run(batch_size, query_len))

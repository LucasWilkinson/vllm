# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the env-gated HiSparse diagnostics."""

import logging
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import (
    HISPARSE_BLOCK_SIZE,
    _publish_hisparse_pages,
    make_hisparse_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.hisparse import debug_stats
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator

BS = 4


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _values(result):
    return {key: int(value) for key, value in result.items()}


def test_classify_topk_resident_host_masked():
    # Row 0: pages [resident 5, host-only, neither]; row 1: all resident.
    resident_bt = torch.tensor([[5, 0, 0], [7, 8, 9]], dtype=torch.int32)
    source_bt = torch.tensor([[1, 2, 0], [3, 4, 6]], dtype=torch.int32)
    logical = torch.tensor(
        [
            [0, 4, 8, -1],  # resident, host, masked, padding
            [1, 5, 9, 2],  # all resident
        ],
        dtype=torch.int32,
    )
    converted = torch.tensor([[10, 20, -1, -1], [1, 2, 3, 4]], dtype=torch.int32)
    out = _values(
        debug_stats.classify_topk(
            logical,
            torch.tensor([0, 1]),
            resident_bt,
            source_bt,
            BS,
            converted=converted,
        )
    )
    assert out == {
        "tokens": 2,
        "requested": 7,
        "resident": 5,
        "host": 1,
        "masked": 1,
        "partial_rows": 1,
        "reqs_partial": 1,
        "reqs": 2,
        "conv_masked": 1,
    }


def test_mirror_audit_flags_stale_host_rows():
    # Two requests, 5 pages each; the last two pages are unsealed.
    resident_bt = torch.tensor([[1, 2, 0, 3, 4], [5, 6, 7, 8, 9]])
    source_bt = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    both, host_only = debug_stats.select_mirror_pages(resident_bt, source_bt, 64)
    assert both.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]
    assert host_only.tolist() == [[0, 2]]
    sampled, _ = debug_stats.select_mirror_pages(resident_bt, source_bt, 2)
    assert sampled.tolist() == [[0, 0], [1, 2]]

    resident = torch.arange(2 * BS * 3, dtype=torch.uint8).view(2, BS, 3)
    host = resident.clone()
    host[1, 2, 0] += 1  # one stale row in page 1
    assert debug_stats.compare_mirror_rows(resident, host) == (1, 1)
    assert debug_stats.compare_mirror_rows(resident, resident.clone()) == (0, 0)
    host[0] = 0
    assert debug_stats.count_zero_rows(host) == BS


def test_label_groups_by_name_then_buffer():
    names = ["model.layers.0.attn", "model.layers.77.attn", "model.layers.78.attn"]
    assert debug_stats.label_groups(names, [1, 1, 2], 78) == [
        "target",
        "target",
        "mtp",
    ]
    # No parseable layer index: the minority logical top-k buffer is MTP.
    assert debug_stats.label_groups(["a", "b", "c"], [1, 1, 2], None) == [
        "target",
        "target",
        "mtp",
    ]


def _fake_handle(logical, resident_bt, source_bt, *, leader=True, top_k=4):
    rows = logical.shape[0]
    shared = SimpleNamespace(
        physical_topk_indices=torch.full((rows, top_k), 7, dtype=torch.int32),
        device_topk_rows=torch.full((rows, top_k), 40, dtype=torch.int32),
        swap_device_physical_rows=torch.full((rows, top_k), 40, dtype=torch.int32),
        swap_counts=torch.tensor([1] * rows, dtype=torch.int32),
    )
    runtime = SimpleNamespace(
        is_group_leader=leader,
        index_group=SimpleNamespace(shared_topk=shared),
        host_cache=torch.zeros(16 * BS, 3, dtype=torch.uint8),
    )
    mla_group = SimpleNamespace(
        logical_topk_indices=logical,
        physical_topk_indices=torch.full((rows, top_k), 7, dtype=torch.int32),
        request_ids=torch.arange(rows, dtype=torch.int32),
    )
    return SimpleNamespace(
        runtime=runtime,
        mla_index_group=mla_group,
        block_table=resident_bt,
        source_block_table=source_bt,
        view=SimpleNamespace(
            block_size=BS, cache=torch.zeros(16, BS, 3, dtype=torch.uint8)
        ),
        debug_attn_metadata=None,
        dummy_batch=False,
        num_actual_tokens=0,
    )


def test_worker_stats_split_target_and_mtp(monkeypatch, caplog):
    monkeypatch.setattr(debug_stats, "INTERVAL", 2)
    resident_bt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32)
    source_bt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32)
    target_logical = torch.tensor([[0, 4, 8, 12]], dtype=torch.int32)
    # MTP reads page 1, which has lost its resident copy.
    mtp_resident_bt = torch.tensor([[1, 0, 3, 4, 5, 6]], dtype=torch.int32)
    mtp_logical = torch.tensor([[0, 4, -1, -1]], dtype=torch.int32)
    target = _fake_handle(target_logical, resident_bt, source_bt)
    mtp = _fake_handle(mtp_logical, mtp_resident_bt, source_bt)
    # MTP page 0 host copy is stale relative to its live resident copy.
    mtp.view.cache[1, 2, 0] = 9
    handles = [target, mtp]
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=1)
        )
    )
    with caplog.at_level(logging.INFO, logger=debug_stats.logger.name):
        stats = debug_stats.WorkerDebugStats(
            config,
            ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"],
            handles,
            is_rank0=True,
        )
        assert stats.labels == ["target", "mtp"]

        metadata = SimpleNamespace(
            num_decode_tokens=1,
            decode_max_query_len=1,
            num_decodes=1,
            req_id_per_token=torch.zeros(1, dtype=torch.int32),
        )
        target.debug_attn_metadata = metadata
        target.num_actual_tokens = 1
        stats.on_target_forward_done(handles)
        stats.on_draft_forward(
            "draft_prefill", {"model.layers.1.self_attn.attn": metadata}, 1
        )
        stats.on_draft_forward("draft_decode", None, 1)
        stats.count_mirror_dma("bulk", [0, 1], 3)
        stats.on_step_start(1)
        assert not any("topk" in record.getMessage() for record in caplog.records)
        stats.on_step_start(1)

    lines = [record.getMessage() for record in caplog.records]
    target_line = next(line for line in lines if "group=target phase=verify" in line)
    assert "requested=4 resident=4 host=0 masked=0" in target_line
    assert "host_miss=1" in target_line
    prefill_line = next(line for line in lines if "phase=draft_prefill" in line)
    assert "requested=2 resident=1 host=1 masked=0" in prefill_line
    assert "host_hit=0" in prefill_line
    assert any("phase=draft_decode" in line for line in lines)
    mirror_mtp = next(line for line in lines if "mirror step=2 group=mtp" in line)
    assert "rows_mismatch=1" in mirror_mtp and "hostonly_pages=1" in mirror_mtp
    mirror_target = next(line for line in lines if "mirror step=2 group=target" in line)
    assert "rows_mismatch=0" in mirror_target
    assert any("mtp_bulk=3" in line and "target_bulk=3" in line for line in lines)
    race = next(line for line in lines if "race step=2 group=target" in line)
    assert "checks=1 rows=1 swaps=1 same_step_dup_claims=0 clobbered_entries=0" in race


def test_detect_slot_conflicts_sequential_vs_racing_rows():
    # One request, 3 spec positions resolved by the per-position loop.
    # Step 0 fills slot 100 (host row 5); step 1 hits it; step 2 fills 101.
    hot_rows = torch.tensor([[100, 7], [100, 7], [101, 7]])
    swaps = torch.tensor([[100, -1], [-1, -1], [101, -1]])
    counts = torch.tensor([1, 0, 1])
    out = _values(
        debug_stats.detect_slot_conflicts(
            hot_rows, swaps, counts, torch.tensor([0, 1, 2])
        )
    )
    assert out["same_step_dup_claims"] == 0 and out["clobbered_entries"] == 0
    # Step 2 evicts slot 100 that steps 0 and 1 still read.
    swaps_clobber = torch.tensor([[100, -1], [-1, -1], [100, -1]])
    hot_clobber = torch.tensor([[100, 7], [100, 7], [100, 7]])
    out = _values(
        debug_stats.detect_slot_conflicts(
            hot_clobber, swaps_clobber, counts, torch.tensor([0, 1, 2])
        )
    )
    assert out["clobbered_entries"] == 2
    # The same three rows resolved in ONE launch both claiming slot 100.
    out = _values(
        debug_stats.detect_slot_conflicts(
            hot_clobber, swaps_clobber, counts, torch.zeros(3, dtype=torch.long)
        )
    )
    assert out["same_step_dup_claims"] == 1 and out["clobbered_entries"] == 0


def test_record_resolve_launch_counts_rows_per_request():
    group = SimpleNamespace(
        debug_launch=torch.zeros(len(debug_stats.LAUNCH_FIELDS), dtype=torch.int64),
        debug_launch_counts=torch.zeros(4, dtype=torch.int64),
    )
    state = torch.tensor([0, 1, 2, -1])
    debug_stats.record_resolve_launch(group, torch.tensor([0, 1, 2, 3]), state)
    debug_stats.record_resolve_launch(group, torch.tensor([0, 0, 0, 1]), state)
    assert dict(zip(debug_stats.LAUNCH_FIELDS, group.debug_launch.tolist())) == {
        "launches": 2,
        "rows": 7,
        "max_rows_per_req": 3,
        "multirow_launches": 1,
        "multirow_req_launches": 1,
    }
    # No buffers (non-rank-0 / disabled): a no-op.
    debug_stats.record_resolve_launch(SimpleNamespace(), torch.tensor([0]), state)


def test_coordinator_counts_unpin_reuse_and_lost_pages(monkeypatch, caplog):
    monkeypatch.setattr(debug_stats, "ENABLED", True)
    monkeypatch.setattr(debug_stats, "INTERVAL", 1)
    debug_stats.SCHED.clear()
    manager = make_hisparse_kv_cache_manager(32, 16)
    tokens = list(range(4 * HISPARSE_BLOCK_SIZE))
    request = make_request("request", tokens, HISPARSE_BLOCK_SIZE, sha256)
    assert manager.allocate_slots(request, num_new_tokens=len(tokens)) is not None
    coordinator = get_hisparse_coordinator(manager)
    for hot_manager in coordinator.hot_managers:
        hot_manager.require_hot(request.request_id)
    assert manager.allocate_slots(request, num_new_tokens=len(tokens)) is not None
    _publish_hisparse_pages(manager)
    assert debug_stats.SCHED["transfer_planned_seal"] > 0
    assert debug_stats.SCHED["transfer_completed"] > 0
    assert debug_stats.SCHED["unpin"] > 0
    pool = manager.block_pool
    pool.get_new_blocks(pool.get_num_free_blocks())
    assert debug_stats.SCHED["reuse_owned"] > 0
    assert debug_stats.SCHED["lose_page"] == debug_stats.SCHED["unpin"]
    assert debug_stats.SCHED["lose_page_imported"] == 0

    with caplog.at_level(logging.INFO, logger=debug_stats.logger.name):
        debug_stats.maybe_log_scheduler(coordinator)
    (line,) = [r.getMessage() for r in caplog.records if "HISPARSE_DBG sched" in r.msg]
    assert "lose_page=" in line and "gpu_free=0" in line
    assert not debug_stats.SCHED


@pytest.mark.parametrize("enabled", [False, True])
def test_disabled_hooks_leave_no_state(monkeypatch, enabled):
    monkeypatch.setattr(debug_stats, "ENABLED", enabled)
    debug_stats.SCHED.clear()
    manager = make_hisparse_kv_cache_manager(32, 16)
    request = make_request(
        "request", list(range(2 * HISPARSE_BLOCK_SIZE)), HISPARSE_BLOCK_SIZE, sha256
    )
    assert manager.allocate_slots(request, num_new_tokens=32) is not None
    _publish_hisparse_pages(manager)
    assert bool(debug_stats.SCHED) == enabled


def test_remirror_knob_copies_draft_rows_before_next_forward(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker as mod
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnectorMetadata,
    )
    from vllm.v1.hisparse.types import SparseKVRowMirror

    names = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
    handles = [SimpleNamespace(mla_index_group=None) for _ in names]
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=1)
        )
    )
    assert debug_stats.draft_layer_indices(config, names, handles) == [1]

    monkeypatch.setattr(mod, "current_stream", MagicMock())
    worker = object.__new__(mod.HiSparseConnectorWorker)
    worker._set_row_mirrors((SparseKVRowMirror((8,), 16, 4),))
    worker._draft_remirror_layers = (1,)
    worker._slot_mapping_staging = None
    worker.host_write_events = (MagicMock(), MagicMock())
    worker.host_write_event = worker.host_write_events[1]
    worker._next_host_write_event = 0
    worker._pending_dma_descriptors = mod.deque()
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker._pending_invalid_block_ids = []
    worker.cache_handles = [
        SimpleNamespace(runtime=SimpleNamespace(eager_host_mirror=True))
    ]
    worker.shared_host_region = None
    calls = []
    worker._enqueue_row_dma = lambda layers, ready_event=None: calls.append(
        (layers, worker._row_mirror_num_rows)
    )
    worker.start_step(HiSparseConnectorMetadata(None, (), (), {}, True), None, [])
    # The previous step's resolved rows were re-mirrored for the draft layer
    # before the new step replaced them.
    assert calls == [((1,), 4)]
    assert worker._row_mirror_num_rows == 0

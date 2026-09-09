# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def _make_speculator(layer_names=("draft.0", "draft.1")) -> tuple[DSparkSpeculator, Mock]:
    speculator = object.__new__(DSparkSpeculator)
    model = Mock()
    model.get_draft_kv_cache_layer_names.return_value = list(layer_names)
    model.combine_hidden_states.side_effect = lambda states: states[:, :2] + 1
    speculator.model = model
    speculator._pcp_context_kv_precomputed = False
    # One KV cache group per draft layer, so build_slot_mappings_by_layer maps
    # row i of the restored slot mappings onto layer i.
    speculator.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=[name]) for name in layer_names
        ]
    )
    return speculator, model


def _manager(states, positions, slot_mappings) -> Mock:
    manager = Mock()
    manager.restore_sharded_context.return_value = (states, positions, slot_mappings)
    return manager


def test_precompute_pcp_context_kv_writes_the_restored_global_batch():
    speculator, model = _make_speculator()
    # The manager all-gathers this rank's shard back to the global batch and
    # returns the rank-local slot mapping over it.
    manager = _manager(
        torch.tensor([[2.0, 4.0], [3.0, 5.0]]),
        torch.tensor([10, 11]),
        torch.tensor([[4, 5], [8, 9]]),
    )
    aux_hidden_states = [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])]

    speculator.precompute_pcp_context_kv(manager, aux_hidden_states)

    model.combine_hidden_states.assert_called_once()
    manager.restore_sharded_context.assert_called_once()
    args, kwargs = model.precompute_and_store_context_kv.call_args
    assert torch.equal(args[0], torch.tensor([[2.0, 4.0], [3.0, 5.0]]))
    assert torch.equal(args[1], torch.tensor([10, 11]))
    assert len(args[2]) == 2
    assert torch.equal(args[2][0], torch.tensor([4, 5]))
    assert torch.equal(args[2][1], torch.tensor([8, 9]))
    assert kwargs == {}
    assert speculator._pcp_context_kv_precomputed

    with pytest.raises(RuntimeError, match="already precomputed"):
        speculator.precompute_pcp_context_kv(manager, aux_hidden_states)


def test_precompute_pcp_context_kv_does_not_retain_for_a_pd_producer():
    speculator, _ = _make_speculator()
    manager = _manager(
        torch.tensor([[2.0, 4.0]]), torch.tensor([10]), torch.tensor([[4], [8]])
    )

    speculator.precompute_pcp_context_kv(
        manager, [torch.tensor([[1.0]]), torch.tensor([[3.0]])],
        retain_for_proposal=False,
    )

    # A PCP prefiller in P/D hands the draft cache to the decoder instead of
    # proposing from it, so the step must not be marked as precomputed.
    assert not speculator._pcp_context_kv_precomputed


def test_precompute_pcp_context_kv_rejects_missing_layer_mapping():
    speculator, _ = _make_speculator()
    # Only one group comes back, so "draft.1" has no slot mapping.
    manager = _manager(
        torch.tensor([[2.0, 4.0]]), torch.tensor([0]), torch.tensor([[0]])
    )

    with pytest.raises(RuntimeError, match="draft.1"):
        speculator.precompute_pcp_context_kv(
            manager, [torch.ones(1, 1), torch.ones(1, 1)]
        )


def test_precompute_pcp_context_kv_requires_aux_states():
    speculator, _ = _make_speculator()
    with pytest.raises(RuntimeError, match="auxiliary hidden states"):
        speculator.precompute_pcp_context_kv(Mock(), [])

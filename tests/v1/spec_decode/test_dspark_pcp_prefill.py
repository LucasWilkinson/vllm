# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def _make_speculator() -> tuple[DSparkSpeculator, Mock]:
    speculator = object.__new__(DSparkSpeculator)
    model = Mock()
    model.get_draft_kv_cache_layer_names.return_value = ["draft.0", "draft.1"]
    model.combine_hidden_states.side_effect = lambda states: states[:, :2] + 1
    speculator.model = model
    speculator._pcp_context_kv_precomputed = False
    return speculator, model


def test_precompute_pcp_context_kv_uses_only_local_rows():
    speculator, model = _make_speculator()
    input_batch = SimpleNamespace(
        num_tokens=2,
        positions=torch.tensor([10, 11, 99]),
    )
    aux_hidden_states = [
        torch.tensor([[1.0], [2.0], [90.0]]),
        torch.tensor([[3.0], [4.0], [91.0]]),
    ]
    slot_mappings = {
        "draft.0": torch.tensor([4, 5, 40]),
        "draft.1": torch.tensor([8, 9, 80]),
    }

    speculator.precompute_pcp_context_kv(input_batch, aux_hidden_states, slot_mappings)

    model.combine_hidden_states.assert_called_once()
    args, kwargs = model.precompute_and_store_context_kv.call_args
    assert torch.equal(args[0], torch.tensor([[2.0, 4.0], [3.0, 5.0]]))
    assert torch.equal(args[1], torch.tensor([10, 11]))
    assert len(args[2]) == 2
    assert torch.equal(args[2][0], torch.tensor([4, 5]))
    assert torch.equal(args[2][1], torch.tensor([8, 9]))
    assert kwargs == {}
    assert speculator._pcp_context_kv_precomputed

    with pytest.raises(RuntimeError, match="already precomputed"):
        speculator.precompute_pcp_context_kv(
            input_batch, aux_hidden_states, slot_mappings
        )


def test_precompute_pcp_context_kv_rejects_missing_layer_mapping():
    speculator, _ = _make_speculator()
    input_batch = SimpleNamespace(num_tokens=1, positions=torch.tensor([0]))

    with pytest.raises(RuntimeError, match="draft.1"):
        speculator.precompute_pcp_context_kv(
            input_batch,
            [torch.ones(1, 1), torch.ones(1, 1)],
            {"draft.0": torch.tensor([0])},
        )


def test_precompute_pcp_context_kv_projects_restored_global_context():
    """The projection is unsharded: ``combine_hidden_states`` runs on the
    restored global context, not on this rank's shard (PR #54036 removed)."""
    speculator, model = _make_speculator()
    input_batch = SimpleNamespace(num_tokens=1, positions=torch.tensor([10]))
    aux_hidden_states = [torch.tensor([[1.0]]), torch.tensor([[3.0]])]
    local_slot_mappings = {"draft.0": torch.tensor([4])}

    global_states = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    global_positions = torch.tensor([10, 11])
    global_slot_mappings = {
        "draft.0": torch.tensor([4, 6]),
        "draft.1": torch.tensor([8, 12]),
    }

    seen = []

    def restore_context(states):
        seen.append(states)
        return global_states, global_positions, global_slot_mappings

    speculator.precompute_pcp_context_kv(
        input_batch,
        aux_hidden_states,
        local_slot_mappings,
        restore_context=restore_context,
    )

    # The restore saw the raw (un-combined) auxiliary states...
    assert len(seen) == 1
    assert torch.equal(seen[0], torch.tensor([[1.0, 3.0]]))
    # ...and the projection ran over the full restored global batch.
    combine_arg = model.combine_hidden_states.call_args[0][0]
    assert torch.equal(combine_arg, global_states)
    args, _ = model.precompute_and_store_context_kv.call_args
    assert torch.equal(args[0], global_states[:, :2] + 1)
    assert torch.equal(args[1], global_positions)
    assert torch.equal(args[2][0], torch.tensor([4, 6]))
    assert torch.equal(args[2][1], torch.tensor([8, 12]))

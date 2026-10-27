# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types

import pytest
import torch

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def nixl_adapter_module(monkeypatch):
    # Only the external transport is mocked; construct the real vLLM adapter.
    from vllm.utils.import_utils import has_nixl_ep

    name = "vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep"
    if name in sys.modules or has_nixl_ep():
        yield importlib.import_module(name)
        return
    transport = types.ModuleType("nixl_ep")
    transport.Buffer = object  # type: ignore[attr-defined]
    transport.topk_idx_t = torch.int64  # type: ignore[attr-defined]
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "nixl_ep", transport)
        module = importlib.import_module(name)
        try:
            yield module
        finally:
            sys.modules.pop(name, None)
            parent = sys.modules[name.rsplit(".", 1)[0]]
            if getattr(parent, "nixl_ep", None) is module:
                delattr(parent, "nixl_ep")


class _RecordingTransport:
    def dispatch(self, inputs, ids, *args, **kwargs):
        self.dispatch_ids = ids.clone()
        return None, None, object(), None, lambda: None

    def combine(self, inputs, ids, *args, **kwargs):
        self.combine_ids = ids.clone()
        return None, None, lambda: None


@pytest.mark.parametrize(
    "ids, expected, round_robin",
    [
        ([[0, 7], [1, 6]], [[0, 7], [2, 5]], True),
        ([[0, -1], [-1, 6]], [[0, -1], [-1, 5]], True),
        ([[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]], True),
        ([[0, -1], [-1, 6]], [[0, -1], [-1, 6]], False),
    ],
    ids=["valid", "mixed-padding", "all-padding", "linear"],
)
def test_nixl_dispatch_and_combine_preserve_invalid_routes(
    nixl_adapter_module, ids, expected, round_robin
):
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceDelegate,
    )

    transport = _RecordingTransport()
    mapping = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7]) if round_robin else None
    adapter = nixl_adapter_module.NixlEPPrepareAndFinalize(
        buffer=transport,
        max_tokens_per_rank=64,
        num_dispatchers=4,
        expert_capacity=8,
        global_to_physical=mapping,
    )
    routes = torch.tensor(ids, dtype=adapter.topk_indices_dtype())
    original_routes = routes.clone()
    inputs = torch.zeros((2, 2048), dtype=torch.bfloat16)
    weights = torch.ones((2, 2), dtype=torch.float32)
    adapter.prepare_async(
        inputs, weights, routes, 8, None, False, FusedMoEQuantConfig.make()
    )
    adapter.finalize_async(
        torch.empty_like(inputs),
        torch.empty((2, 256, 2048), dtype=inputs.dtype),
        weights,
        routes,
        False,
        TopKWeightAndReduceDelegate(),
    )
    reference = torch.tensor(expected, dtype=routes.dtype)
    torch.testing.assert_close(transport.dispatch_ids, reference)
    torch.testing.assert_close(transport.combine_ids, reference)
    torch.testing.assert_close(routes, original_routes)

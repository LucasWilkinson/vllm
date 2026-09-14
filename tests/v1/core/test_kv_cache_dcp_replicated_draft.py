# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A replicated speculative draft's KV must not be scaled by the DCP size."""

import torch

from vllm.v1.core.kv_cache_utils import (
    dcp_world_size_for_kv_cache_spec,
    resolve_dcp_kv_block_size,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _spec(**kw):
    return FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=64, dtype=torch.bfloat16, **kw
    )


def test_target_layers_are_sharded_by_default():
    assert resolve_dcp_kv_block_size(_spec(), 8) == 128
    assert dcp_world_size_for_kv_cache_spec(_spec(), 8) == 8


def test_replicated_draft_keeps_its_block_geometry():
    spec = _spec(dcp_sharded=False)
    assert resolve_dcp_kv_block_size(spec, 8) == 16
    assert dcp_world_size_for_kv_cache_spec(spec, 8) == 1


def test_no_dcp_is_unaffected():
    assert resolve_dcp_kv_block_size(_spec(dcp_sharded=False), 1) == 16
    assert resolve_dcp_kv_block_size(_spec(), 1) == 16

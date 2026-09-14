# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DFlash/DSpark draft collapses the PCP axis; DCP must collapse with it."""

import pytest

from vllm.config import ParallelConfig
from vllm.config.utils import replace
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import draft_parallel_config


@pytest.mark.parametrize(
    ("tp", "pcp", "dcp", "expected_dcp"),
    [
        (1, 8, 8, 1),  # DCP spans the PCP axis (GLM-5.3 PCP8+DCP8+EP8, TP1)
        (1, 4, 4, 1),
        (2, 4, 8, 2),  # DCP spans TP x PCP
        (8, 8, 64, 8),
        (1, 8, 1, 1),  # DCP off
    ],
)
def test_draft_collapses_dcp_with_pcp(tp, pcp, dcp, expected_dcp):
    target = ParallelConfig(
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
    )
    draft = draft_parallel_config(target)
    assert draft.prefill_context_parallel_size == 1
    assert draft.decode_context_parallel_size == expected_dcp
    # The derived config must satisfy the pcp == 1 branch of the validator.
    assert draft.tensor_parallel_size % draft.decode_context_parallel_size == 0


def test_resetting_pcp_alone_is_rejected():
    # The bug: this target is valid, and resetting only pcp trips
    # ParallelConfig's "tp_size must be divisible by dcp_size" check.
    target = ParallelConfig(
        tensor_parallel_size=1,
        prefill_context_parallel_size=8,
        decode_context_parallel_size=8,
    )
    with pytest.raises(ValueError, match="must be divisible by dcp_size"):
        replace(target, prefill_context_parallel_size=1)


def test_without_pcp_is_passthrough():
    target = ParallelConfig(tensor_parallel_size=8, decode_context_parallel_size=2)
    assert draft_parallel_config(target) is target

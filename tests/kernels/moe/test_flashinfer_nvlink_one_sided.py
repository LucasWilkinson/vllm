# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    flashinfer_nvlink_one_sided,
)

_GET_FORWARD_CONTEXT = (
    "vllm.model_executor.layers.fused_moe.prepare_finalize."
    "flashinfer_nvlink_one_sided.get_forward_context"
)


def test_get_local_sizes_without_dp_metadata():
    context = SimpleNamespace(dp_metadata=None)
    with patch(_GET_FORWARD_CONTEXT, return_value=context):
        assert flashinfer_nvlink_one_sided.get_local_sizes(4, 2) == [4, 4]


def test_get_local_sizes_with_dp_metadata():
    metadata = SimpleNamespace(local_sizes=[3, 5])
    context = SimpleNamespace(dp_metadata=metadata)
    with patch(_GET_FORWARD_CONTEXT, return_value=context):
        assert flashinfer_nvlink_one_sided.get_local_sizes(4, 2) == [3, 5]


def test_get_local_sizes_with_dp_metadata_outside_chunked_context():
    metadata = SimpleNamespace(local_sizes=None)
    context = SimpleNamespace(dp_metadata=metadata)
    with patch(_GET_FORWARD_CONTEXT, return_value=context):
        assert flashinfer_nvlink_one_sided.get_local_sizes(4, 2) == [4, 4]

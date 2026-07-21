# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared high-priority CUDA side streams."""

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream

_streams: dict[int, torch.cuda.Stream] = {}


def get_side_stream() -> torch.cuda.Stream | None:
    """Return the current device's shared high-priority CUDA stream."""
    if not current_platform.is_cuda():
        return None

    main_stream = current_stream()
    device_index = main_stream.device.index
    assert device_index is not None
    stream = _streams.get(device_index)
    if stream is None:
        _, high_priority = torch.cuda.Stream.priority_range()
        stream = torch.cuda.Stream(device=device_index, priority=high_priority)
        _streams[device_index] = stream
    return stream


def wait_side_stream() -> None:
    """Make the current stream wait for work issued on the side stream.

    No-op on platforms without a CUDA side stream.
    """
    main_stream = current_stream()
    stream = get_side_stream()
    if stream is not None:
        main_stream.wait_stream(stream)


def register_side_stream() -> None:
    """Register streams used by PyTorch's compiled stream operators."""
    from torch._dynamo.graph_bytecode_inputs import (
        CURRENT_STREAM_INDEX,
        index_to_external_object_weakref,
        set_external_object_by_index,
    )

    main_stream = current_stream()
    side_stream = get_side_stream()
    if side_stream is None:
        return

    for index, stream in (
        (CURRENT_STREAM_INDEX, main_stream),
        (CURRENT_STREAM_INDEX + 1, side_stream),
    ):
        stream_ref = index_to_external_object_weakref.get(index)
        if stream_ref is None or stream_ref() is not stream:
            set_external_object_by_index(index, stream)


def _patch_external_object_getter() -> None:
    import torch._dynamo.graph_bytecode_inputs as graph_inputs

    getter = graph_inputs.get_external_object_by_index
    if getattr(getter, "_vllm_side_stream_patched", False):
        return

    def get_external_object_by_index(index: int):
        if index not in index_to_external_object_weakref and index in (0, 1):  # type: ignore[name-defined]  # noqa: F821
            _vllm_register_side_stream()  # type: ignore[name-defined]  # noqa: F821
        if index not in index_to_external_object_weakref:  # type: ignore[name-defined]  # noqa: F821
            raise AssertionError("Index not registered in index_to_user_object_weakref")
        obj = index_to_external_object_weakref[index]()  # type: ignore[name-defined]  # noqa: F821
        if obj is None:
            raise AssertionError("User object is no longer alive")
        return obj

    graph_inputs._vllm_register_side_stream = register_side_stream
    getter.__code__ = get_external_object_by_index.__code__
    getter._vllm_side_stream_patched = True


_patch_external_object_getter()

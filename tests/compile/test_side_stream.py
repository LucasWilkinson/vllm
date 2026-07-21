# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from contextlib import contextmanager

import pytest
import torch
from torch._dynamo.testing import EagerAndRecordGraphs

import vllm.compilation.side_stream as side_stream
from vllm.compilation.backends import graph_uses_stream_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.side_stream import get_side_stream
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.envs import disable_envs_cache
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_torch_equal_or_newer


@contextmanager
def use_vllm_config(vllm_config: VllmConfig):
    with set_forward_context({}, vllm_config), set_current_vllm_config(vllm_config):
        yield


@support_torch_compile
class NativeSideStreamModule(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.side_stream = get_side_stream()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.side_stream is not None
        self.side_stream.wait_stream(torch.accelerator.current_stream())
        with self.side_stream:
            side = x + 1
        main = x * 2
        torch.accelerator.current_stream().wait_stream(self.side_stream)
        return main + side


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only test")
def test_side_stream_uses_native_compile_context() -> None:
    stream = get_side_stream()
    assert stream is not None
    backend = EagerAndRecordGraphs()

    def run(x: torch.Tensor) -> torch.Tensor:
        stream.wait_stream(torch.accelerator.current_stream())
        with stream:
            side = x + 1
        torch.accelerator.current_stream().wait_stream(stream)
        return side

    x = torch.zeros(4, device="cuda")
    actual = torch.compile(run, backend=backend, fullgraph=True)(x)
    assert torch.equal(actual, x + 1)
    assert len(backend.graphs) == 1
    assert graph_uses_stream_ops(backend.graphs[0])

    annotated_nodes = [
        node
        for node in backend.graphs[0].graph.nodes
        if node.meta.get("custom", {}).get("stream") not in (None, 0)
    ]
    assert annotated_nodes
    assert all(
        "vllm.side_stream" not in str(node.target)
        for node in backend.graphs[0].graph.nodes
    )
    wait_stream_nodes = [
        node
        for node in backend.graphs[0].graph.nodes
        if "streams.wait_stream" in str(node.target)
    ]
    assert len(wait_stream_nodes) == 2


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only test")
@pytest.mark.skipif(not is_torch_equal_or_newer("2.13.0"), reason="requires torch 2.13")
def test_side_stream_aot_cache_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as cache_dir, monkeypatch.context() as m:
        m.setenv("VLLM_CACHE_ROOT", cache_dir)
        m.setenv("VLLM_USE_AOT_COMPILE", "1")
        m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
        m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
        disable_envs_cache()

        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                backend="inductor",
            )
        )
        x = torch.randn(16, device="cuda")
        expected = 3 * x + 1
        with use_vllm_config(vllm_config):
            compiled_module = NativeSideStreamModule(vllm_config=vllm_config)
            torch.testing.assert_close(compiled_module(x), expected)

        disable_envs_cache()
        m.setenv("VLLM_FORCE_AOT_LOAD", "1")
        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                backend="inductor",
            )
        )
        with use_vllm_config(vllm_config):
            cached_module = NativeSideStreamModule(vllm_config=vllm_config)
            from torch._dynamo.graph_bytecode_inputs import reset_user_object_tracking

            reset_user_object_tracking()
            side_stream._streams.clear()
            torch.testing.assert_close(cached_module(x), expected)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = cached_module(x)
            graph.replay()
            torch.testing.assert_close(captured, expected)

        assert cached_module.was_aot_compile_fn_loaded_from_disk
        disable_envs_cache()

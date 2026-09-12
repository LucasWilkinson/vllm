# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
)
from vllm.triton_utils import tl, triton


def canonicalize_topk(indices: torch.Tensor) -> None:
    """Order selected token IDs in place, with padding last."""
    _CANONICALIZE_TOPK_KERNEL(indices, topk=indices.shape[1])


class CanonicalizeTopkKernel(VllmTritonJitKernel["CanonicalizeTopkKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        topk: int

    @staticmethod
    @triton.jit(do_not_specialize=["row_stride", "col_stride"])
    def kernel(indices, row_stride, col_stride, TOPK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, TOPK)
        offsets = row * row_stride + cols * col_stride
        values = tl.load(indices + offsets)
        values = tl.sort(tl.where(values < 0, 0x7FFFFFFF, values))
        tl.store(indices + offsets, tl.where(values == 0x7FFFFFFF, -1, values))

    def dispatch(self, *, topk: int) -> CompileKey:  # type: ignore[override]
        return self.CompileKey(topk=topk)

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        topk = getattr(vllm_config.model_config.hf_config, "index_topk", 0)
        return [self.CompileKey(topk=topk)] if topk in (512, 1024, 2048) else []

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            indices=TritonWarmupTensor(torch.int32, shape=(1, compile_key.topk)),
            topk=compile_key.topk,
        )

    @kernel_launcher
    def __call__(self, indices: torch.Tensor, *, topk: int) -> LaunchSpec:
        return (indices.shape[0],), dict(
            row_stride=indices.stride(0),
            col_stride=indices.stride(1),
            TOPK=topk,
            num_warps=max(4, topk // 64),
        )


_CANONICALIZE_TOPK_KERNEL = CanonicalizeTopkKernel()

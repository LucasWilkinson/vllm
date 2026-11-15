# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import Any

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


def mamba1_tp_parts(
    shapes: tuple[tuple[int, ...], ...], conv_dim: int
) -> tuple[tuple[int, tuple[int, ...]], ...] | None:
    """`MambaSpec.tp_parts` of the conv state (`x` at dim `conv_dim`) and the
    temporal state `[x, state]`, both sharded across TP along `x`."""
    if len(shapes) != 2:
        return None
    conv, temporal = shapes
    return (conv_dim, (conv[conv_dim],)), (0, (temporal[0],))


class Mamba1AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA1_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True

    @classmethod
    def customize_mamba_spec(cls, spec: MambaSpec) -> MambaSpec:
        conv_dim = 0 if is_conv_state_dim_first() else 1
        return replace(spec, tp_parts=mamba1_tp_parts(spec.shapes, conv_dim))


@dataclass
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata):
    pass


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    metadata_cls = Mamba1AttentionMetadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba1AttentionMetadata:
        return self._compute_common_metadata(common_attn_metadata)

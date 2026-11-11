# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deprecated KV cache spec API: policy-typed specs, `get_kv_cache_spec` and
`KVCacheSpecRegistry`, translated to `KVCacheLayerConfig`. Remove next release.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    CircularBufferManager,
    CrossAttentionManager,
    EncoderOnlyAttentionManager,
    FullAttentionManager,
    MambaManager,
    RSWAManager,
    SingleTypeKVCacheManager,
    SinkFullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayerConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase


@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    sliding_window: int | None = None
    attention_chunk_size: int | None = None
    non_causal: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.non_causal:
            raise ValueError(
                "FullAttentionSpec(non_causal=True) is no longer supported; "
                "disable chunked prefill and prefix caching in the model config."
            )


@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int
    extra_retained_tokens: int = 0


@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(MLAAttentionSpec, SlidingWindowSpec):
    bounded_replay: bool = False


@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int


@dataclass(frozen=True, kw_only=True)
class CircularBufferSpec(AttentionSpec):
    pass


@dataclass(frozen=True, kw_only=True)
class RSWASpec(FullAttentionSpec):
    rswa_window: int


@dataclass(frozen=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    sink_len: int | None = None


@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    pass


@dataclass(frozen=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    pass


_DEPRECATED_SPECS = (
    FullAttentionSpec,
    SlidingWindowSpec,
    SlidingWindowMLASpec,
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    RSWASpec,
    SinkFullAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
)

_SPEC_MANAGERS: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    AttentionSpec: FullAttentionManager,
    MambaSpec: MambaManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    CircularBufferSpec: CircularBufferManager,
    RSWASpec: RSWAManager,
    SinkFullAttentionSpec: SinkFullAttentionManager,
    CrossAttentionSpec: CrossAttentionManager,
    EncoderOnlyAttentionSpec: EncoderOnlyAttentionManager,
}


class KVCacheSpecRegistry:
    """Deprecated: return `Manager.layer_config(spec, **params)` from the layer's
    `get_kv_cache_config` instead of registering a spec class."""

    @classmethod
    def register(
        cls,
        kvcache_spec_cls: type[KVCacheSpec],
        manager_class: type[SingleTypeKVCacheManager] | None = None,
        uniform_type_base_spec: type[KVCacheSpec] | None = None,
    ) -> None:
        """Map `kvcache_spec_cls` to `manager_class`; `uniform_type_base_spec` is
        ignored since layers now group by manager config."""
        warnings.warn(
            "KVCacheSpecRegistry is deprecated; return "
            "`Manager.layer_config(spec, **params)` from `get_kv_cache_config`.",
            DeprecationWarning,
            stacklevel=2,
        )
        assert manager_class is not None, "manager_class is required"
        registered = _SPEC_MANAGERS.setdefault(kvcache_spec_cls, manager_class)
        assert registered is manager_class, (
            f"Conflicting registration for KVCacheSpec: {kvcache_spec_cls.__name__}"
        )

    @classmethod
    def get_manager_class(cls, spec: KVCacheSpec) -> type[SingleTypeKVCacheManager]:
        for base in type(spec).__mro__:
            if base in _SPEC_MANAGERS:
                return _SPEC_MANAGERS[base]
        raise ValueError(f"No KV cache manager registered for {type(spec).__name__}")


def register_kv_cache_spec(
    manager_class: type[SingleTypeKVCacheManager] | None = None,
    uniform_type_base_spec: type[KVCacheSpec] | None = None,
):
    def decorator(kvcache_spec_cls: type[KVCacheSpec]) -> type[KVCacheSpec]:
        KVCacheSpecRegistry.register(
            kvcache_spec_cls, manager_class, uniform_type_base_spec
        )
        return kvcache_spec_cls

    return decorator


def get_layer_kv_cache_config(
    layer: AttentionLayerBase, vllm_config: VllmConfig
) -> KVCacheLayerConfig | None:
    """`layer.get_kv_cache_config`, or its deprecated `get_kv_cache_spec`
    translated when the layer still defines one.

    The manager comes from the spec's class, its `Params` from the spec's fields
    of the same name. Deprecated spec classes are narrowed to their layout base;
    other (custom) classes stay the layer's spec as is.
    """
    get_kv_cache_spec = getattr(layer, "get_kv_cache_spec", None)
    if get_kv_cache_spec is None:
        return layer.get_kv_cache_config(vllm_config)
    warnings.warn(
        f"{type(layer).__name__}.get_kv_cache_spec is deprecated; implement "
        "get_kv_cache_config returning `Manager.layer_config(spec, **params)`.",
        DeprecationWarning,
        stacklevel=2,
    )
    from vllm.platforms import current_platform

    current_platform.register_custom_kv_cache_specs(vllm_config)
    spec = get_kv_cache_spec(vllm_config)
    if spec is None:
        return None
    manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
    params = {
        f.name: getattr(spec, f.name)
        for f in fields(manager_cls.Params)
        if hasattr(spec, f.name)
    }
    layout_cls = next(c for c in type(spec).__mro__ if c not in _DEPRECATED_SPECS)
    if layout_cls is not type(spec):
        spec = layout_cls(
            **{f.name: getattr(spec, f.name) for f in fields(layout_cls) if f.init}
        )
    return manager_cls.layer_config(spec, **params)

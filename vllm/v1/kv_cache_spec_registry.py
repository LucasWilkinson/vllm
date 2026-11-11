# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deprecated: see `vllm.v1.kv_cache_compat`."""

from vllm.v1.kv_cache_compat import KVCacheSpecRegistry, register_kv_cache_spec

__all__ = ["KVCacheSpecRegistry", "register_kv_cache_spec"]

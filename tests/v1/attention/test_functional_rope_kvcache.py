# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

import vllm.config
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    dense_kv_cache_views,
)
from vllm.compilation.passes.fusion.rope_kvcache_fusion import (
    RopeKVCacheFusionPass,
)
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ModelConfig,
    PassConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.rotary_embedding import (
    BailingMRotaryEmbedding,
    MRotaryEmbedding,
    MRotaryEmbeddingInterleaved,
    RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
    Ernie4_5_VLRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.fope import FourierRotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheLayout

_LAYER_NAME = "model.layers.0.self_attn.attn"
_MROPE_SECTION = [8, 12, 12]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA backend constructors")
@pytest.mark.parametrize(
    "selection,enabled,is_cuda,supports_fusion",
    [
        ("auto", True, True, False),
        ("override", True, True, False),
        ("auto", False, True, False),
        ("auto", True, False, False),
        ("auto", True, True, True),
        ("flash", True, True, True),
    ],
)
def test_fuse_rope_kvcache_gate_uses_selected_backend_capability(
    selection: str,
    enabled: bool,
    is_cuda: bool,
    supports_fusion: bool,
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config: VllmConfig,
) -> None:
    from vllm.model_executor.layers.attention import attention as attention_module

    backend = (
        AttentionBackendEnum.FLASH_ATTN
        if selection == "flash"
        else AttentionBackendEnum.TRITON_ATTN
    ).get_class()
    default_vllm_config.compilation_config.pass_config.fuse_rope_kvcache = enabled
    default_vllm_config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
    monkeypatch.setattr(current_platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(attention_module, "get_attn_backend", lambda *a, **kw: backend)
    # A future supporting backend must not be rejected by its name.
    if selection != "flash" and supports_fusion:
        monkeypatch.setattr(
            backend.get_impl_cls(), "fused_rope_kvcache_q_out_supported", lambda _: True
        )

    with set_default_torch_dtype(torch.float16):
        layer = Attention(
            num_heads=4,
            head_size=64,
            scale=0.125,
            num_kv_heads=2,
            prefix=_LAYER_NAME,
            attn_backend=backend if selection == "override" else None,
        )
    assert layer.attn_backend is backend
    assert layer.impl.fused_rope_kvcache_q_out_supported() is supports_fusion
    assert layer._fuse_rope_kvcache is (enabled and supports_fusion)


def _rope(dtype: torch.dtype = torch.float16) -> RotaryEmbedding:
    return RotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=128,
        base=10000,
        is_neox_style=True,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    ("enabled", "num_tokens", "fused"),
    [(True, 256, True), (True, 257, False), (False, 1, False)],
)
def test_fused_rope_rotation_applies_gate_and_token_threshold(
    enabled: bool,
    num_tokens: int,
    fused: bool,
    default_vllm_config: VllmConfig,
) -> None:
    layer = SimpleNamespace(
        _fuse_rope_kvcache=enabled, rope_kvcache_fusion_max_token_num=256
    )
    positions = torch.arange(num_tokens)
    query = torch.randn(num_tokens, 64, dtype=torch.float16)

    rotation = Attention._get_fused_rope_rotation(layer, positions, query, _rope())

    assert (rotation is not None) is fused


def test_rotary_embedding_rotation_indexes_its_cache(
    default_vllm_config: VllmConfig,
) -> None:
    rotary_emb = _rope()
    positions = torch.arange(4)
    query = torch.randn(4, 64, dtype=torch.float16)

    rotation = rotary_emb.get_rotation(positions, query.dtype)

    assert rotation is not None
    assert rotation.positions is positions
    assert rotation.cos_sin is rotary_emb.cos_sin_cache
    assert rotation.is_neox
    assert rotary_emb.get_rotation(positions.expand(3, -1), query.dtype) is None


def test_fourier_rotary_embedding_is_not_fusable() -> None:
    vllm_config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["none"]))
    with vllm.config.set_current_vllm_config(vllm_config):
        rotary_emb = FourierRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=128,
            base=10000,
            is_neox_style=True,
            dtype=torch.float16,
            init_cache=False,
            num_key_value_heads=2,
            num_inv_freq=32,
            fope_sep_head=True,
            fope_init_factor=1.0,
        )

    assert rotary_emb.get_rotation(torch.arange(4), torch.float16) is None


def _make_mrope(kind: str, is_neox: bool) -> MRotaryEmbedding:
    args = (64, 64, 512, 10000, is_neox, torch.float32)
    if kind == "mrope":
        return MRotaryEmbedding(*args, mrope_section=_MROPE_SECTION)
    if kind == "mrope-interleaved":
        return MRotaryEmbedding(
            *args, mrope_section=_MROPE_SECTION, mrope_interleaved=True
        )
    if kind == "interleaved":
        return MRotaryEmbeddingInterleaved(*args, mrope_section=_MROPE_SECTION)
    if kind == "ernie":
        return Ernie4_5_VLRotaryEmbedding(*args, mrope_section=[12, 12, 8])
    assert kind == "bailing"
    return BailingMRotaryEmbedding(*args, mrope_section=[8, 12, 12])


@pytest.mark.parametrize(
    "kind", ["mrope", "mrope-interleaved", "interleaved", "ernie", "bailing"]
)
@pytest.mark.parametrize("positions_ndim", [1, 2])
@pytest.mark.parametrize("is_neox", [True, False])
def test_mrope_rotation_matches_unfused_forward(
    kind: str,
    positions_ndim: int,
    is_neox: bool,
    default_vllm_config: VllmConfig,
) -> None:
    """A fused kernel applying `get_rotation` must match the layer's forward."""
    if kind == "interleaved" and positions_ndim == 1:
        pytest.skip("MRotaryEmbeddingInterleaved requires T/H/W positions")
    torch.manual_seed(0)
    num_tokens = 7
    rotary_emb = _make_mrope(kind, is_neox)
    positions = torch.randint(0, 512, (3, num_tokens))
    if positions_ndim == 1:
        positions = positions[0]
    query = torch.randn(num_tokens, 4 * 64)
    key = torch.randn(num_tokens, 2 * 64)

    rope_positions, cos_sin, rope_is_neox = rotary_emb.get_rotation(
        positions, query.dtype
    )
    assert rope_is_neox is is_neox
    if rope_positions is None:
        rope_positions = torch.arange(num_tokens)
    actual = RotaryEmbedding.forward_static(
        rope_positions, query, key, 64, 64, cos_sin, is_neox
    )
    forward = (
        rotary_emb.forward
        if isinstance(rotary_emb, MRotaryEmbeddingInterleaved)
        else rotary_emb.forward_native
    )
    expected = forward(positions, query.clone(), key.clone())

    torch.testing.assert_close(actual, expected)


def test_missing_slot_mapping_rotates_query_without_materializing_key(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm import _custom_ops

    calls = []
    monkeypatch.setattr(
        _custom_ops,
        "rotary_embedding",
        lambda _positions, _query, key, head_size, *_args, **_kwargs: (
            calls.append((key, head_size))
        ),
    )

    query = torch.randn(1, 4 * 64)
    key = torch.randn(1, 2 * 64)
    value = torch.randn_like(key)
    layer = SimpleNamespace(
        impl=SimpleNamespace(fused_rope_kvcache_q_out_supported=lambda: True),
        rope_kvcache_fusion_max_token_num=256,
        head_size=64,
    )
    query_out = torch.empty_like(query, memory_format=torch.contiguous_format)
    Attention._rope_and_kv_cache_update_q_out(
        layer,
        query,
        key,
        value,
        query_out,
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, 1),
        True,
        torch.empty(1),
        None,
    )

    torch.testing.assert_close(query_out, query)
    assert query_out.data_ptr() != query.data_ptr()
    assert calls == [(None, 64)]


class _FunctionalRoPEAttention(torch.nn.Module):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, mrope: bool):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_size = 4, 2, 64
        self.qkv_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_size
        self.qkv_proj = torch.nn.Linear(
            self.qkv_size, self.qkv_size, bias=False, dtype=torch.float16
        )
        rope_args = (self.head_size, self.head_size, 128, 10000, True, torch.float16)
        self.rotary_emb = (
            MRotaryEmbedding(*rope_args, mrope_section=_MROPE_SECTION)
            if mrope
            else RotaryEmbedding(*rope_args)
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.head_size**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix=_LAYER_NAME,
            attn_backend=AttentionBackendEnum.FLASH_ATTN.get_class(),
        )
        self.attn._k_scale = self.attn._k_scale.to(device)
        self.attn._v_scale = self.attn._v_scale.to(device)
        self.backend = self.attn.get_attn_backend()

    def _split_qkv(self, hidden_states: torch.Tensor):
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        qkv = self.qkv_proj(hidden_states)
        return qkv.split([q_size, kv_size, kv_size], dim=-1)

    def incumbent(self, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        query, key, value = self._split_qkv(qkv)
        query, key = self.rotary_emb(positions, query, key)
        return self.attn(query, key, value)

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        query, key, value = self._split_qkv(qkv)
        return self.attn(
            query, key, value, positions=positions, rotary_emb=self.rotary_emb
        )


def _positions(num_tokens: int, mrope: bool) -> torch.Tensor:
    positions = torch.arange(num_tokens, dtype=torch.long, device="cuda")
    # Distinct T/H/W rows so section selection matters.
    return (
        torch.stack([positions, positions * 2, positions * 3]) if mrope else positions
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA.")
@pytest.mark.parametrize("mrope", [False, True])
def test_q_out_rope_kvcache_stays_before_attention_with_graph_owned_output(
    mrope: bool, mocker, disable_vllm_compile_cache, tmp_path
):
    from vllm.compilation.backends import VllmBackend

    dtype = torch.float16
    device = torch.device("cuda")
    num_tokens = 2
    model_dir = tmp_path / "model"
    Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(model_dir)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(model_dir), tokenizer=str(model_dir), dtype=dtype
        ),
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=num_tokens,
            max_num_seqs=num_tokens,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            use_inductor_graph_partition=False,
            inductor_compile_config={"force_disable_caches": True},
            pass_config=PassConfig(
                fuse_rope_kvcache=True,
                fuse_attn_quant=False,
            ),
        ),
    )
    vllm_config.cache_config.kv_cache_layout = KVCacheLayout.LBNHC.name
    assert "vllm::unified_attention_with_output" in (
        vllm_config.compilation_config.splitting_ops or []
    )

    with (
        torch.device(device),
        set_default_torch_dtype(dtype),
        vllm.config.set_current_vllm_config(vllm_config),
    ):
        torch.manual_seed(0)
        model = _FunctionalRoPEAttention(vllm_config, device, mrope)
        assert model.attn._fuse_rope_kvcache
        qkv = torch.randn(
            num_tokens,
            (model.num_heads + 2 * model.num_kv_heads) * model.head_size,
            dtype=dtype,
        )
        positions = _positions(num_tokens, mrope)
        common_metadata = create_common_attn_metadata(
            BatchSpec([num_tokens], [num_tokens]),
            block_size=16,
            device=device,
            arange_block_indices=True,
        )
        cache_spec = model.attn.get_kv_cache_spec(vllm_config)
        assert cache_spec is not None
        builder = model.backend.get_builder_cls()(
            cache_spec,
            [_LAYER_NAME],
            vllm_config,
            device,
        )
        metadata = builder.build(0, common_metadata)
        cache_storage = torch.zeros(
            cache_spec.page_size_bytes, dtype=torch.int8, device=device
        )
        cache = dense_kv_cache_views(
            cache_storage,
            cache_spec,
            num_blocks=1,
            num_layers=1,
            layout=KVCacheLayout.LBNHC,
        )[0]

        def run(call):
            model.attn.kv_cache = cache.clone()
            with set_forward_context(
                metadata,
                vllm_config,
                slot_mapping={_LAYER_NAME: metadata.slot_mapping},
            ):
                output = call(qkv, positions)
            return output, model.attn.kv_cache.clone()

        incumbent_output, incumbent_cache = run(model.incumbent)
        fused_update = mocker.spy(model.attn.impl, "do_rope_and_kv_cache_update_q_out")
        torch._dynamo.mark_dynamic(qkv, 0)
        torch._dynamo.mark_dynamic(positions, positions.dim() - 1)
        backend = VllmBackend(vllm_config)
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        fused_output, fused_cache = run(compiled)

    # CUDA owns this fusion at the model call site, so the legacy ROCm graph
    # pass must not also be registered.
    fusion_passes = [
        pass_
        for pass_ in backend.pass_manager.passes
        if isinstance(pass_, RopeKVCacheFusionPass)
    ]
    assert fusion_passes == []
    call_nodes = [
        node for node in backend.graph.graph.nodes if node.op == "call_function"
    ]
    fused_nodes = [
        node
        for node in call_nodes
        if node.target is torch.ops.vllm.fused_rope_and_unified_kv_cache_update_q_out
    ]
    attention_nodes = [
        node
        for node in call_nodes
        if node.target is torch.ops.vllm.unified_attention_with_output
    ]
    assert len(fused_nodes) == len(attention_nodes) == 1
    assert fused_nodes[0].args[-1] is attention_nodes[0].args[4]
    assert attention_nodes[0].args[1:3] == (None, None)
    fused_update.assert_called_once()
    torch.testing.assert_close(incumbent_output, fused_output, atol=2e-3, rtol=2e-3)
    assert torch.count_nonzero(fused_cache).item() > 0
    torch.testing.assert_close(incumbent_cache, fused_cache, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA.")
@pytest.mark.parametrize("token_counts", [(3, 2), (2, 3)])
@pytest.mark.parametrize("mrope", [False, True])
def test_compiled_manual_rope_runtime_threshold_tracks_dynamic_tokens(
    mrope: bool,
    token_counts: tuple[int, int],
    mocker,
    disable_vllm_compile_cache,
    tmp_path,
) -> None:
    from vllm.compilation.backends import VllmBackend

    dtype = torch.float16
    device = torch.device("cuda")
    threshold = 2
    block_size = 16
    model_dir = tmp_path / "model"
    Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(model_dir)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(model_dir), tokenizer=str(model_dir), dtype=dtype
        ),
        cache_config=CacheConfig(block_size=block_size, cache_dtype="auto"),
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=max(token_counts),
            max_num_seqs=max(token_counts),
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
            use_inductor_graph_partition=False,
            inductor_compile_config={"force_disable_caches": True},
            pass_config=PassConfig(
                fuse_rope_kvcache=True,
                fuse_attn_quant=False,
                rope_kvcache_fusion_max_token_num=threshold,
            ),
        ),
    )
    vllm_config.cache_config.kv_cache_layout = KVCacheLayout.LBNHC.name

    with (
        torch.device(device),
        set_default_torch_dtype(dtype),
        vllm.config.set_current_vllm_config(vllm_config),
    ):
        torch.manual_seed(0)
        model = _FunctionalRoPEAttention(vllm_config, device, mrope)
        assert model.attn._fuse_rope_kvcache
        cache_spec = model.attn.get_kv_cache_spec(vllm_config)
        assert cache_spec is not None
        builder = model.backend.get_builder_cls()(
            cache_spec,
            [_LAYER_NAME],
            vllm_config,
            device,
        )
        fused_update = mocker.spy(model.attn.impl, "do_rope_and_kv_cache_update_q_out")
        fallback_update = mocker.spy(model.attn.impl, "do_kv_cache_update")
        backend = VllmBackend(vllm_config)
        compiled = torch.compile(model, backend=backend, fullgraph=True)

        def run(call, qkv: torch.Tensor, positions: torch.Tensor):
            num_tokens = qkv.shape[0]
            common_metadata = create_common_attn_metadata(
                BatchSpec([num_tokens], [num_tokens]),
                block_size=block_size,
                device=device,
                arange_block_indices=True,
            )
            metadata = builder.build(0, common_metadata)
            num_blocks = (num_tokens + block_size - 1) // block_size
            cache_storage = torch.zeros(
                num_blocks * cache_spec.page_size_bytes,
                dtype=torch.int8,
                device=device,
            )
            model.attn.kv_cache = dense_kv_cache_views(
                cache_storage,
                cache_spec,
                num_blocks=num_blocks,
                num_layers=1,
                layout=KVCacheLayout.LBNHC,
            )[0]
            with set_forward_context(
                metadata,
                vllm_config,
                slot_mapping={_LAYER_NAME: metadata.slot_mapping},
            ):
                output = call(qkv, positions)
            return output, model.attn.kv_cache.clone()

        for index, num_tokens in enumerate(token_counts):
            qkv = torch.randn(num_tokens, model.qkv_size, dtype=dtype, device=device)
            positions = _positions(num_tokens, mrope)
            expected_output, expected_cache = run(model.incumbent, qkv, positions)
            fused_update.reset_mock()
            fallback_update.reset_mock()
            if index == 0:
                torch._dynamo.mark_dynamic(qkv, 0)
                torch._dynamo.mark_dynamic(positions, positions.dim() - 1)
            actual_output, actual_cache = run(compiled, qkv, positions)

            assert fused_update.call_count == int(num_tokens <= threshold)
            assert fallback_update.call_count == int(num_tokens > threshold)
            torch.testing.assert_close(
                actual_output, expected_output, atol=2e-3, rtol=2e-3
            )
            torch.testing.assert_close(
                actual_cache, expected_cache, atol=2e-3, rtol=2e-3
            )

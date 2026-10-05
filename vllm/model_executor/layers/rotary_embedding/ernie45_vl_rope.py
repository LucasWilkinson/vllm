# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from .mrope import MRotaryEmbedding


class Ernie4_5_VLRotaryEmbedding(MRotaryEmbedding):
    """3D rotary positional embedding. 3D is t:time h:height w:width."""

    def _select_cos_sin(
        self, positions: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin_cache[positions].chunk(2, dim=-1)
        if positions.ndim == 1:
            return cos, sin

        assert self.mrope_section
        section_h, section_w, section_t = self.mrope_section
        assert section_h == section_w

        def select(x: torch.Tensor) -> torch.Tensor:
            # Split according to [h w h w h w h w... t t t...]
            x_t = x[0, :, -section_t:]
            x_h = x[1, :, : section_h + section_w : 2]
            x_w = x[2, :, 1 : section_h + section_w : 2]
            x_hw = torch.stack([x_h, x_w], dim=-1).flatten(-2)
            return torch.cat([x_hw, x_t], dim=-1)

        return select(cos), select(sin)

    def forward_native(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        num_tokens = positions.shape[-1]
        cos, sin = self._select_cos_sin(positions, query)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self.apply_rotary_emb.forward_native(
            query_rot,
            cos,
            sin,
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self.apply_rotary_emb.forward_native(
            key_rot,
            cos,
            sin,
        )
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key)

    def forward_xpu(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # No fused XPU kernel for this 3D t/h/w rope; base
        # MRotaryEmbedding.forward_xpu forwards an extra `offsets` arg that
        # this class's forward_cuda override doesn't accept. Use native path.
        return self.forward_native(positions, query, key)

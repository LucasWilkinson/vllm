# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from .base import RopeRotation
from .mrope import MRotaryEmbedding


class Ernie4_5_VLRotaryEmbedding(MRotaryEmbedding):
    """3D rotary positional embedding. 3D is t:time h:height w:width."""

    def get_rotation(self, positions: torch.Tensor, dtype: torch.dtype) -> RopeRotation:
        cos_sin = self._cos_sin_cache_as(dtype, positions.device)[positions]
        if positions.ndim == 2:
            assert self.mrope_section
            section_h, section_w, section_t = self.mrope_section
            assert section_h == section_w
            # Split cos and sin according to [h w h w h w h w... t t t...]
            cos_sin = cos_sin.view(*cos_sin.shape[:-1], 2, -1)
            hw = torch.stack(
                [
                    cos_sin[1, ..., : section_h + section_w : 2],
                    cos_sin[2, ..., 1 : section_h + section_w : 2],
                ],
                dim=-1,
            ).flatten(-2)
            cos_sin = torch.cat([hw, cos_sin[0, ..., -section_t:]], dim=-1)
            cos_sin = cos_sin.flatten(-2)
        return RopeRotation(cos_sin, None, self.is_neox_style)

    def forward_native(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        num_tokens = positions.shape[-1]
        cos, sin = self.get_rotation(positions, query.dtype).cos_sin.chunk(2, dim=-1)

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

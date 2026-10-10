# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numba ports of the block table, staged write and KV zeroing Triton
kernels. Pointer arrays hold host addresses, read with `int32_array`."""

import torch

from vllm.v1.worker.cpu.kernels.utils import int32_array, numba_kernel


@numba_kernel
def gather_block_tables(
    grid,
    batch_idx_to_req_idx,
    src_block_table_ptrs,
    dst_block_table_ptrs,
    block_table_strides,
    num_blocks_ptr,
    num_blocks_stride,
    num_reqs,
    BLOCK_SIZE,
):
    num_groups, num_reqs_padded = grid
    max_num_reqs = num_blocks_ptr.shape[1]
    for group_id in range(num_groups):
        stride = block_table_strides[group_id]
        src = int32_array(src_block_table_ptrs[group_id], (max_num_reqs, stride))
        dst = int32_array(dst_block_table_ptrs[group_id], (num_reqs_padded, stride))
        dst[num_reqs:] = 0
        for batch_idx in range(num_reqs):
            req = batch_idx_to_req_idx[batch_idx]
            num_blocks = num_blocks_ptr[group_id, req]
            dst[batch_idx, :num_blocks] = src[req, :num_blocks]


@numba_kernel
def compute_slot_mappings(
    grid,
    max_num_tokens,
    idx_mapping,
    query_start_loc,
    pos,
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    kernel_block_sizes,
    slot_mapping_enabled,
    dcp_sharded,
    slot_mappings_ptr,
    slot_mappings_stride,
    cp_rank,
    CP_SIZE,
    CP_INTERLEAVE,
    PAD_ID,
    TRITON_BLOCK_SIZE,
):
    num_groups, num_reqs = grid[0], grid[1] - 1
    num_rows = idx_mapping[:num_reqs].max() + 1 if num_reqs > 0 else 0
    for group_id in range(num_groups):
        slot_mapping = slot_mappings_ptr[group_id]
        slot_mapping[:max_num_tokens] = PAD_ID
        if not slot_mapping_enabled[group_id]:
            continue
        block_table = int32_array(
            block_table_ptrs[group_id], (num_rows, block_table_strides[group_id])
        )
        block_size = kernel_block_sizes[group_id]
        kv_block_size = block_sizes[group_id]
        virtual_block_size = kv_block_size * CP_SIZE
        sharded = CP_SIZE != 1 and dcp_sharded[group_id]
        for batch_idx in range(num_reqs):
            req = idx_mapping[batch_idx]
            # A negative idx_mapping entry is a dummy request without blocks.
            if req < 0:
                continue
            for t in range(query_start_loc[batch_idx], query_start_loc[batch_idx + 1]):
                local_pos = pos[t]
                if sharded:
                    offset = local_pos % virtual_block_size
                    if offset // CP_INTERLEAVE % CP_SIZE != cp_rank:
                        continue
                    local_pos = (
                        local_pos // virtual_block_size * kv_block_size
                        + offset // (CP_INTERLEAVE * CP_SIZE) * CP_INTERLEAVE
                        + offset % CP_INTERLEAVE
                    )
                block_number = block_table[req, local_pos // block_size]
                slot_mapping[t] = block_number * block_size + local_pos % block_size


def apply_write(
    output_ptr: torch.Tensor,
    output_stride: int | torch.Tensor,
    write_indices_ptr: torch.Tensor,
    write_starts_ptr: torch.Tensor,
    write_contents_ptr: torch.Tensor,
    write_cu_lens_ptr: torch.Tensor,
    write_group_ids_ptr: torch.Tensor | None,
    BLOCK_SIZE: int,
    MULTI_GROUP: bool,
    *,
    grid: tuple[int],
) -> None:
    args = (write_indices_ptr, write_starts_ptr, write_contents_ptr, write_cu_lens_ptr)
    if MULTI_GROUP:
        _apply_multi_group_write(
            output_ptr, output_stride, write_group_ids_ptr, *args, grid=grid
        )
    else:
        _apply_write(output_ptr.view(-1), output_stride, *args, grid=grid)


@numba_kernel
def _apply_write(grid, output, stride, indices, starts, contents, cu_lens):
    for i in range(grid[0]):
        content = contents[cu_lens[i - 1] if i > 0 else 0 : cu_lens[i]]
        start = indices[i] * stride + starts[i]
        output[start : start + len(content)] = content


@numba_kernel
def _apply_multi_group_write(
    grid, output_ptrs, strides, group_ids, indices, starts, contents, cu_lens
):
    for i in range(grid[0]):
        content = contents[cu_lens[i - 1] if i > 0 else 0 : cu_lens[i]]
        stride = strides[group_ids[i]]
        output = int32_array(output_ptrs[group_ids[i]], ((indices[i] + 1) * stride,))
        start = indices[i] * stride + starts[i]
        output[start : start + len(content)] = content


@numba_kernel
def zero_kv_blocks(
    grid,
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    BLOCK_SIZE,
):
    # Strides and page sizes are in int32 elements.
    for block_id in block_ids_ptr:
        for seg in range(len(seg_addrs_ptr)):
            start = block_id * seg_block_strides_ptr[seg]
            end = start + seg_page_sizes_ptr[seg]
            int32_array(seg_addrs_ptr[seg], (end,))[start:] = 0


@numba_kernel
def dcp_local_seq_lens(
    grid,
    out_ptr,
    seq_lens_ptr,
    dcp_size,
    dcp_rank,
    cp_interleave,
    num_reqs,
    max_num_reqs,
    BLOCK_SIZE,
):
    out_ptr[num_reqs:max_num_reqs] = 0
    for i in range(num_reqs):
        # Distribute KV cache among different ranks, in a round-robin manner.
        seq_len = seq_lens_ptr[i]
        rounds = seq_len // (dcp_size * cp_interleave)
        remainder = seq_len % (dcp_size * cp_interleave) - dcp_rank * cp_interleave
        out_ptr[i] = rounds * cp_interleave + min(max(remainder, 0), cp_interleave)

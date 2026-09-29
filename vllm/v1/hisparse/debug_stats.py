# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated HiSparse diagnostics (``VLLM_HISPARSE_DEBUG_STATS=1``).

Debug-only. Every hook is behind the module constant ``ENABLED``, so an unset
env var costs one global bool check per hook. Worker-side counters accumulate
on the GPU without host syncs; the periodic log (every
``VLLM_HISPARSE_DEBUG_INTERVAL`` steps, TP rank 0 only) synchronizes.

Log lines (grep ``HISPARSE_DBG``):

* ``HISPARSE_DBG topk group=<target|mtp> phase=<verify|draft_prefill|
  draft_decode>`` -- per-phase top-k residency breakdown.
* ``HISPARSE_DBG mirror group=<target|mtp>`` -- host copy vs live resident copy
  of the same sealed page (both present), and all-zero host rows on host-only
  pages. Row mismatches on a clean resident page mean the host mirror is stale.
* ``HISPARSE_DBG mirror_dma`` -- rows DMA'd to host per group and path
  (``bulk`` = end of target forward, ``layer`` = inside a non-FULL forward).
* ``HISPARSE_DBG race`` -- hot-slot conflicts within one forward: a slot
  claimed twice in the same resolve step (concurrent rows), or a slot an earlier
  spec step reads that a later step overwrites before attention runs.
* ``HISPARSE_DBG launch`` -- rows per request inside one
  ``hisparse_resolve_residency`` launch (1 = per-position loop, >1 = rows of
  one request resolved concurrently). Recorded by ops captured into CUDA graphs,
  so FULL-graph replays are counted too.
* ``HISPARSE_DBG sched`` -- coordinator residency events, pool gauges and
  scheduler preemption / allocation-failure counters.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.getenv("VLLM_HISPARSE_DEBUG_STATS", "0") == "1"
INTERVAL = max(int(os.getenv("VLLM_HISPARSE_DEBUG_INTERVAL", "200")), 1)
MIRROR_PAGES = max(int(os.getenv("VLLM_HISPARSE_DEBUG_MIRROR_PAGES", "64")), 0)
# Experimental A/B knob, independent of ENABLED: re-mirror the draft (MTP)
# layers' rows of the previous step at the next step start, after the drafter
# has written them. Without it, a FULL-graph target step DMAs the MTP rows at
# the end of the target forward, before the drafter writes them.
REMIRROR_DRAFT = os.getenv("VLLM_HISPARSE_DEBUG_REMIRROR_DRAFT", "0") == "1"
# Run the (syncing) hot-slot race check every N forwards; 0 disables it.
RACE_EVERY = max(int(os.getenv("VLLM_HISPARSE_DEBUG_RACE_EVERY", "1")), 0)
# Pages this close to a request's last host page may still be written.
_UNSEALED_TAIL_PAGES = 2

# Scheduler-process event counters (coordinator + scheduler increment these).
SCHED: Counter[str] = Counter()
_sched_calls = 0

# Worker-process singleton, set by HiSparseConnectorWorker.initialize.
WORKER: WorkerDebugStats | None = None

TOPK_FIELDS = (
    "tokens",
    "requested",
    "resident",
    "host",
    "masked",
    "conv_masked",
    "host_miss",
    "partial_rows",
    "reqs_partial",
    "reqs",
    "all_conv_neg",
    "all_host_miss",
    "all_logical_neg",
)
RACE_FIELDS = ("checks", "rows", "swaps", "same_step_dup_claims", "clobbered_entries")
LAUNCH_FIELDS = (
    "launches",
    "rows",
    "max_rows_per_req",
    "multirow_launches",
    "multirow_req_launches",
)


# ---------------------------------------------------------------------------
# Pure helpers (CPU-testable)
# ---------------------------------------------------------------------------


def classify_topk(
    logical: torch.Tensor,
    token_rows: torch.Tensor,
    resident_bt: torch.Tensor,
    source_bt: torch.Tensor,
    block_size: int,
    converted: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Mirror phase 1 of ``hisparse_resolve_residency`` on the host side.

    ``logical``: [T, K] request-relative token positions (-1 = not requested).
    ``token_rows``: [T] block-table row per token. Resident page 0 / host page
    <= 0 are null. Returns 0-dim int64 tensors (no host sync).
    """
    logical = logical.long()
    num_reqs = int(resident_bt.shape[0])
    rows = token_rows.long().clamp(min=0, max=max(num_reqs - 1, 0))
    requested = logical >= 0
    page = torch.where(requested, logical // block_size, 0)

    def lookup(bt: torch.Tensor) -> torch.Tensor:
        cols = bt.shape[1]
        in_range = page < cols
        values = bt.long()[rows[:, None], page.clamp(max=max(cols - 1, 0))]
        return torch.where(in_range & requested, values, 0)

    resident = requested & (lookup(resident_bt) > 0)
    host = requested & ~resident & (lookup(source_bt) > 0)
    masked = requested & ~resident & ~host
    masked_per_row = masked.sum(dim=1)
    requested_per_row = requested.sum(dim=1)
    partial = (masked_per_row > 0) & (masked_per_row < requested_per_row)
    per_req = torch.zeros((2, num_reqs), dtype=torch.int64, device=logical.device)
    per_req[0].index_add_(0, rows, partial.long())
    per_req[1].index_add_(0, rows, (requested_per_row > 0).long())
    out = {
        "tokens": torch.tensor(logical.shape[0], device=logical.device),
        "requested": requested.sum(),
        "resident": resident.sum(),
        "host": host.sum(),
        "masked": masked.sum(),
        "partial_rows": partial.sum(),
        "reqs_partial": (per_req[0] > 0).sum(),
        "reqs": (per_req[1] > 0).sum(),
    }
    if converted is not None:
        out["conv_masked"] = (requested & (converted[: logical.shape[0]] < 0)).sum()
    return {key: value.to(torch.int64) for key, value in out.items()}


def select_mirror_pages(
    resident_bt: torch.Tensor,
    source_bt: torch.Tensor,
    max_pages: int,
    tail_pages: int = _UNSEALED_TAIL_PAGES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick sealed pages to audit. CPU block tables in, [P, 2] (row, page) out.

    Returns (pages with both a resident and a host block, host-only pages).
    """
    cols = min(resident_bt.shape[1], source_bt.shape[1])
    resident_bt = resident_bt[:, :cols]
    source_bt = source_bt[:, :cols]
    has_host = source_bt > 0
    page_ids = torch.arange(cols)
    last = torch.where(has_host, page_ids[None, :], -1).max(dim=1).values
    sealed = page_ids[None, :] <= (last - tail_pages)[:, None]
    both = (resident_bt > 0) & has_host & sealed
    host_only = (resident_bt <= 0) & has_host & sealed

    def sample(mask: torch.Tensor) -> torch.Tensor:
        pairs = mask.nonzero()
        if pairs.shape[0] > max_pages > 0:
            keep = torch.linspace(0, pairs.shape[0] - 1, max_pages).round().long()
            pairs = pairs[keep]
        elif max_pages == 0:
            pairs = pairs[:0]
        return pairs

    return sample(both), sample(host_only)


def compare_mirror_rows(
    resident_pages: torch.Tensor,
    host_pages: torch.Tensor,
) -> tuple[int, int]:
    """[P, B, W] resident vs host page bytes -> (mismatched rows, pages)."""
    if resident_pages.numel() == 0:
        return 0, 0
    row_diff = (resident_pages != host_pages).flatten(2).any(dim=-1)
    return int(row_diff.sum()), int(row_diff.any(dim=-1).sum())


def count_zero_rows(host_pages: torch.Tensor) -> int:
    if host_pages.numel() == 0:
        return 0
    return int((host_pages == 0).flatten(2).all(dim=-1).sum())


def detect_slot_conflicts(
    hot_rows: torch.Tensor,
    swap_device_rows: torch.Tensor,
    swap_counts: torch.Tensor,
    row_steps: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Find hot slots whose content differs from what a resolved row expects.

    ``hot_rows`` [R, K]: physical row each top-k entry reads (resident or hot,
    -1 masked). ``swap_device_rows`` [R, K] with the first ``swap_counts[r]``
    entries valid: hot rows each resolve row filled from host. ``row_steps``
    [R]: resolve launch order of each row (spec position for the per-position
    loop, 0 when all rows share one launch). Hot rows are per request, so a
    physical row identifies (request, slot).

    * ``same_step_dup_claims``: (slot, step) pairs filled more than once in one
      launch -- two rows raced for one slot and one of them reads foreign KV.
    * ``clobbered_entries``: entries reading a slot that a later launch of the
      same forward refills before attention consumes it.
    """
    device = hot_rows.device
    num_rows, top_k = swap_device_rows.shape
    valid = torch.arange(top_k, device=device)[None, :] < swap_counts[:, None]
    steps = row_steps.long()[:, None].expand(num_rows, top_k)
    valid &= swap_device_rows >= 0
    dev = swap_device_rows[valid].long()
    dev_steps = steps[valid]
    zero = torch.zeros((), dtype=torch.int64, device=device)
    out = {
        "checks": torch.ones((), dtype=torch.int64, device=device),
        "rows": torch.tensor(num_rows, dtype=torch.int64, device=device),
        "swaps": torch.tensor(dev.numel(), dtype=torch.int64, device=device),
        "same_step_dup_claims": zero,
        "clobbered_entries": zero,
    }
    if dev.numel() == 0:
        return out
    num_steps = int(row_steps.max().item()) + 1
    _, claims = torch.unique(dev * num_steps + dev_steps, return_counts=True)
    out["same_step_dup_claims"] = (claims > 1).sum()
    slots, inverse = torch.unique(dev, return_inverse=True)
    last_step = torch.full_like(slots, -1).scatter_reduce(
        0, inverse, dev_steps, reduce="amax"
    )
    entries = hot_rows.long()
    position = torch.searchsorted(slots, entries).clamp(max=slots.numel() - 1)
    found = (entries >= 0) & (slots[position] == entries)
    later = last_step[position] > row_steps.long()[:, None]
    out["clobbered_entries"] = (found & later).sum()
    return out


def record_resolve_launch(
    group: Any, request_rows: torch.Tensor, request_state_indices: torch.Tensor
) -> None:
    """Count rows per request in one resolve launch (CUDA-graph capturable)."""
    acc = getattr(group, "debug_launch", None)
    if acc is None:
        return
    counts = group.debug_launch_counts
    rows = request_rows.long().clamp(0, counts.numel() - 1)
    active = request_state_indices.index_select(0, rows) >= 0
    counts.zero_()
    counts.index_add_(0, rows, active.long())
    peak = counts.max()
    acc[0] += 1
    acc[1] += active.sum()
    acc[2] = torch.maximum(acc[2], peak)
    acc[3] += (peak > 1).long()
    acc[4] += (counts > 1).sum()


def label_groups(
    layer_names: Sequence[str],
    logical_ptrs: Sequence[int | None],
    num_target_layers: int | None,
) -> list[str]:
    """Label each index-group leader ``target`` or ``mtp``.

    Primary: layer index >= the target's num_hidden_layers is an MTP layer.
    Fallback: the MTP module owns its own logical top-k buffer, so leaders not
    sharing the majority buffer are MTP.
    """
    ptr_counts = Counter(ptr for ptr in logical_ptrs if ptr is not None)
    majority = ptr_counts.most_common(1)[0][0] if ptr_counts else None
    labels = []
    for name, ptr in zip(layer_names, logical_ptrs, strict=True):
        match = re.search(r"layers\.(\d+)\.", name)
        if match is not None and num_target_layers:
            labels.append(
                "mtp" if int(match.group(1)) >= num_target_layers else "target"
            )
        elif ptr is not None and majority is not None:
            labels.append("target" if ptr == majority else "mtp")
        else:
            labels.append("target")
    return labels


def draft_layer_indices(
    vllm_config: Any, layer_names: Sequence[str], cache_handles: Sequence[Any]
) -> list[int]:
    hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
    ptrs = [
        (
            handle.mla_index_group.logical_topk_indices.data_ptr()
            if getattr(handle, "mla_index_group", None) is not None
            else None
        )
        for handle in cache_handles
    ]
    labels = label_groups(
        layer_names, ptrs, getattr(hf_config, "num_hidden_layers", None)
    )
    return [index for index, label in enumerate(labels) if label == "mtp"]


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _fmt(values: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in values.items())


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


class WorkerDebugStats:
    def __init__(
        self,
        vllm_config: Any,
        layer_names: Sequence[str],
        cache_handles: Sequence[Any],
        is_rank0: bool,
    ) -> None:
        self.is_rank0 = is_rank0
        self.handles = list(cache_handles)
        self.layer_names = list(layer_names)
        hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
        num_target_layers = getattr(hf_config, "num_hidden_layers", None)
        ptrs = [
            (
                handle.mla_index_group.logical_topk_indices.data_ptr()
                if getattr(handle, "mla_index_group", None) is not None
                else None
            )
            for handle in self.handles
        ]
        self.labels = label_groups(self.layer_names, ptrs, num_target_layers)
        self.leaders = [
            (index, label)
            for index, (handle, label) in enumerate(
                zip(self.handles, self.labels, strict=True)
            )
            if handle.runtime.is_group_leader
        ]
        target_leaders = [i for i, label in self.leaders if label == "target"]
        self.last_target_leader = target_leaders[-1] if target_leaders else None
        target_layers = [i for i, lb in enumerate(self.labels) if lb == "target"]
        mtp_layers = [i for i, lb in enumerate(self.labels) if lb == "mtp"]
        # Audit every MTP layer plus the first and last target layers.
        self.audit_layers = sorted(
            {*mtp_layers, *target_layers[:1], *target_layers[-1:]}
        )
        self.topk: dict[tuple[str, str], torch.Tensor] = {}
        self.race: dict[tuple[str, str], torch.Tensor] = {}
        self.forwards: Counter[str] = Counter()
        if is_rank0:
            for index, _ in self.leaders:
                group = self.handles[index].runtime.index_group
                state = getattr(
                    self.handles[index].runtime, "request_state_indices", None
                )
                if state is None or getattr(group, "debug_launch", None) is not None:
                    continue
                group.debug_launch = torch.zeros(
                    len(LAUNCH_FIELDS), dtype=torch.int64, device=state.device
                )
                group.debug_launch_counts = torch.zeros(
                    state.numel(), dtype=torch.int64, device=state.device
                )
        self.topk_steps: Counter[tuple[str, str]] = Counter()
        self.dma_rows: Counter[str] = Counter()
        self.skipped: Counter[str] = Counter()
        self.steps = 0
        if is_rank0:
            logger.info(
                "HISPARSE_DBG groups target_leaders=%s mtp_leaders=%s "
                "num_target_layers=%s interval=%d",
                [self.layer_names[i] for i, lb in self.leaders if lb == "target"],
                [self.layer_names[i] for i, lb in self.leaders if lb == "mtp"],
                num_target_layers,
                INTERVAL,
            )

    # -- top-k accounting ---------------------------------------------------

    def _accumulate(self, key: tuple[str, str], values: dict[str, torch.Tensor]):
        device = next(iter(values.values())).device
        acc = self.topk.get(key)
        if acc is None:
            acc = torch.zeros(len(TOPK_FIELDS), dtype=torch.int64, device=device)
            self.topk[key] = acc
        for field, value in values.items():
            acc[TOPK_FIELDS.index(field)] += value
        self.topk_steps[key] += 1

    @staticmethod
    def _kernel_outputs(
        handle: Any, num_tokens: int, query_len: int, num_decodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(converted indices, per-step swap counts) left by the last resolve."""
        mla_group = handle.mla_index_group
        shared = handle.runtime.index_group.shared_topk
        if query_len <= 1:
            converted = shared.physical_topk_indices[:num_tokens]
            swap_rows = num_tokens
        else:
            converted = mla_group.physical_topk_indices[:num_tokens]
            swap_rows = query_len * num_decodes
        swap_rows = min(swap_rows, shared.swap_counts.shape[0])
        return converted, shared.swap_counts[:swap_rows]

    def _record_group(
        self,
        label: str,
        phase: str,
        handle: Any,
        token_rows: torch.Tensor,
        num_tokens: int,
        query_len: int,
        num_decodes: int,
        extra: dict[str, torch.Tensor] | None = None,
    ) -> None:
        mla_group = handle.mla_index_group
        if (
            mla_group is None
            or handle.block_table is None
            or handle.source_block_table is None
            or handle.view is None
        ):
            self.skipped[f"{label}_{phase}_unbound"] += 1
            return
        converted, swaps = self._kernel_outputs(
            handle, num_tokens, query_len, num_decodes
        )
        values = classify_topk(
            mla_group.logical_topk_indices[:num_tokens],
            token_rows[:num_tokens],
            handle.block_table,
            handle.source_block_table,
            handle.view.block_size,
            converted=converted,
        )
        values["host_miss"] = swaps.sum().to(torch.int64)
        if extra:
            values.update(extra)
        self._accumulate((label, phase), values)

    def _race_check(
        self,
        label: str,
        phase: str,
        handle: Any,
        num_tokens: int,
        query_len: int,
        num_decodes: int,
    ) -> None:
        shared = handle.runtime.index_group.shared_topk
        if query_len <= 1:
            num_rows, row_steps = num_tokens, None
        elif num_tokens == query_len * num_decodes:
            # Per-position loop: rows are [step0 reqs..., step1 reqs..., ...].
            num_rows = num_tokens
            row_steps = torch.arange(num_rows, device=shared.swap_counts.device)
            row_steps = row_steps // max(num_decodes, 1)
        else:
            self.skipped[f"{label}_{phase}_race_ragged"] += 1
            return
        if num_rows > shared.swap_counts.shape[0]:
            self.skipped[f"{label}_{phase}_race_overflow"] += 1
            return
        if row_steps is None:
            row_steps = torch.zeros(
                num_rows, dtype=torch.int64, device=shared.swap_counts.device
            )
        values = detect_slot_conflicts(
            shared.device_topk_rows[:num_rows],
            shared.swap_device_physical_rows[:num_rows],
            shared.swap_counts[:num_rows],
            row_steps,
        )
        acc = self.race.get((label, phase))
        if acc is None:
            acc = torch.zeros(
                len(RACE_FIELDS), dtype=torch.int64, device=row_steps.device
            )
            self.race[(label, phase)] = acc
        acc += torch.stack([values[field].to(torch.int64) for field in RACE_FIELDS])

    def _race_due(self, key: str) -> bool:
        self.forwards[key] += 1
        return RACE_EVERY > 0 and self.forwards[key] % RACE_EVERY == 0

    def on_target_forward_done(self, cache_handles: Sequence[Any]) -> None:
        """Called at the end of the target forward, before mirror state clears."""
        if not self.is_rank0 or _capturing():
            return
        if self.last_target_leader is None:
            return
        last = cache_handles[self.last_target_leader]
        metadata = getattr(last, "debug_attn_metadata", None)
        if metadata is None or last.dummy_batch or last.num_actual_tokens == 0:
            return
        num_tokens = int(getattr(metadata, "num_decode_tokens", 0))
        if num_tokens == 0:
            self.skipped["target_verify_no_decode"] += 1
            return
        query_len = int(getattr(metadata, "decode_max_query_len", 1) or 1)
        num_decodes = int(getattr(metadata, "num_decodes", num_tokens))
        conv_neg = []
        misses = []
        for index, label in self.leaders:
            if label != "target":
                continue
            converted, swaps = self._kernel_outputs(
                cache_handles[index], num_tokens, query_len, num_decodes
            )
            conv_neg.append((converted < 0).sum())
            misses.append(swaps.sum())
        num_groups = len(conv_neg)
        logical_neg = (last.mla_index_group.logical_topk_indices[:num_tokens] < 0).sum()
        extra = {
            "all_conv_neg": torch.stack(conv_neg).sum().to(torch.int64),
            "all_host_miss": torch.stack(misses).sum().to(torch.int64),
            "all_logical_neg": (logical_neg * num_groups).to(torch.int64),
        }
        self._record_group(
            "target",
            "verify",
            last,
            metadata.req_id_per_token,
            num_tokens,
            query_len,
            num_decodes,
            extra,
        )
        if self._race_due("target"):
            for index, label in self.leaders:
                if label == "target":
                    self._race_check(
                        "target",
                        "verify",
                        cache_handles[index],
                        num_tokens,
                        query_len,
                        num_decodes,
                    )

    def on_draft_forward(
        self, phase: str, attn_metadata: dict[str, Any] | None, num_reqs: int
    ) -> None:
        if not self.is_rank0 or _capturing():
            return
        for index, label in self.leaders:
            if label != "mtp":
                continue
            handle = self.handles[index]
            if phase == "draft_prefill":
                metadata = None
                if attn_metadata is not None:
                    metadata = attn_metadata.get(self.layer_names[index])
                if metadata is None:
                    metadata = getattr(handle, "debug_attn_metadata", None)
                if metadata is None:
                    self.skipped["mtp_draft_prefill_no_metadata"] += 1
                    continue
                num_tokens = int(getattr(metadata, "num_decode_tokens", 0))
                if num_tokens == 0:
                    self.skipped["mtp_draft_prefill_no_decode"] += 1
                    continue
                query_len = int(getattr(metadata, "decode_max_query_len", 1) or 1)
                num_decodes = int(getattr(metadata, "num_decodes", num_tokens))
                token_rows = metadata.req_id_per_token
            else:
                num_tokens = num_decodes = num_reqs
                query_len = 1
                token_rows = handle.mla_index_group.request_ids
            self._record_group(
                "mtp", phase, handle, token_rows, num_tokens, query_len, num_decodes
            )
            if self._race_due(f"mtp_{phase}"):
                self._race_check(
                    "mtp", phase, handle, num_tokens, query_len, num_decodes
                )

    # -- mirror accounting --------------------------------------------------

    def count_mirror_dma(self, path: str, layer_indices: Sequence[int], rows: int):
        for index in layer_indices:
            self.dma_rows[f"{self.labels[index]}_{path}"] += rows

    def audit_mirror(self, num_reqs: int) -> dict[str, Counter[str]]:
        """Compare host rows with live resident rows of sealed pages."""
        results: dict[str, Counter[str]] = {}
        if num_reqs <= 0 or MIRROR_PAGES == 0:
            return results
        for index in self.audit_layers:
            handle = self.handles[index]
            if (
                handle.view is None
                or handle.block_table is None
                or handle.source_block_table is None
            ):
                continue
            resident_bt = handle.block_table[:num_reqs].cpu()
            source_bt = handle.source_block_table[:num_reqs].cpu()
            both, host_only = select_mirror_pages(resident_bt, source_bt, MIRROR_PAGES)
            resident = handle.view.cache
            block_size = resident.shape[1]
            host = handle.runtime.host_cache.view(-1, block_size, resident.shape[-1])
            stats = results.setdefault(self.labels[index], Counter())
            stats["layers"] += 1
            if both.numel():
                res_ids = resident_bt[both[:, 0], both[:, 1]].long()
                host_ids = source_bt[both[:, 0], both[:, 1]].long()
                res_pages = resident[res_ids.to(resident.device)].cpu()
                host_pages = host[host_ids]
                rows, pages = compare_mirror_rows(res_pages, host_pages)
                stats["pages_checked"] += int(both.shape[0])
                stats["rows_checked"] += int(both.shape[0]) * block_size
                stats["rows_mismatch"] += rows
                stats["pages_mismatch"] += pages
                stats["host_zero_rows_resident_pages"] += count_zero_rows(host_pages)
            if host_only.numel():
                host_ids = source_bt[host_only[:, 0], host_only[:, 1]].long()
                stats["hostonly_pages"] += int(host_only.shape[0])
                stats["hostonly_zero_rows"] += count_zero_rows(host[host_ids])
        return results

    # -- periodic log -------------------------------------------------------

    def on_step_start(self, num_reqs: int) -> None:
        self.steps += 1
        if not self.is_rank0 or self.steps % INTERVAL:
            return
        if _capturing():
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for (label, phase), acc in sorted(self.topk.items()):
            values = dict(zip(TOPK_FIELDS, acc.tolist()))
            values["host_hit"] = values["host"] - values["host_miss"]
            if label == "target":
                values["all_conv_masked"] = (
                    values["all_conv_neg"] - values["all_logical_neg"]
                )
            else:
                for field in ("all_conv_neg", "all_host_miss", "all_logical_neg"):
                    values.pop(field)
            logger.info(
                "HISPARSE_DBG topk step=%d group=%s phase=%s steps=%d %s",
                self.steps,
                label,
                phase,
                self.topk_steps[(label, phase)],
                _fmt(values),
            )
            acc.zero_()
        self.topk_steps.clear()
        for (label, phase), acc in sorted(self.race.items()):
            logger.info(
                "HISPARSE_DBG race step=%d group=%s phase=%s %s",
                self.steps,
                label,
                phase,
                _fmt(dict(zip(RACE_FIELDS, acc.tolist()))),
            )
            acc.zero_()
        launches: dict[str, list[int]] = {}
        for index, label in self.leaders:
            acc = getattr(self.handles[index].runtime.index_group, "debug_launch", None)
            if acc is None:
                continue
            values = acc.tolist()
            total = launches.setdefault(label, [0] * len(LAUNCH_FIELDS))
            for field, value in enumerate(values):
                total[field] = (
                    max(total[field], value)
                    if LAUNCH_FIELDS[field] == "max_rows_per_req"
                    else total[field] + value
                )
            acc.zero_()
        for label, values in sorted(launches.items()):
            logger.info(
                "HISPARSE_DBG launch step=%d group=%s %s",
                self.steps,
                label,
                _fmt(dict(zip(LAUNCH_FIELDS, values))),
            )
        for label, stats in sorted(self.audit_mirror(num_reqs).items()):
            logger.info(
                "HISPARSE_DBG mirror step=%d group=%s %s",
                self.steps,
                label,
                _fmt(dict(sorted(stats.items()))),
            )
        logger.info(
            "HISPARSE_DBG mirror_dma step=%d %s skipped=%s",
            self.steps,
            _fmt(dict(sorted(self.dma_rows.items()))),
            dict(self.skipped),
        )
        self.dma_rows.clear()
        self.skipped.clear()


def on_draft_forward(
    phase: str, attn_metadata: dict[str, Any] | None, num_reqs: int
) -> None:
    if WORKER is not None:
        WORKER.on_draft_forward(phase, attn_metadata, num_reqs)


# ---------------------------------------------------------------------------
# Scheduler side
# ---------------------------------------------------------------------------


def maybe_log_scheduler(coordinator: Any) -> None:
    global _sched_calls
    _sched_calls += 1
    if _sched_calls % INTERVAL:
        return
    gauges: dict[str, Any] = {}
    gpu_pool = getattr(coordinator, "gpu_pool", None)
    if gpu_pool is not None:
        gauges["gpu_free"] = gpu_pool.get_num_free_blocks()
        gauges["gpu_total"] = gpu_pool.num_gpu_blocks
        gauges["gpu_usage"] = round(gpu_pool.get_usage(), 4)
    host_manager = getattr(coordinator, "host_manager", None)
    if host_manager is not None:
        gauges["host_free"] = host_manager.block_pool.get_num_free_blocks()
    states = coordinator.request_states.values()
    gauges["reqs"] = len(coordinator.request_states)
    gauges["valid_pages"] = sum(len(s.valid_pages) for s in states)
    gauges["pinned_clean"] = sum(len(s.pinned_clean) for s in states)
    gauges["unpinned"] = sum(len(s.unpinned_pages) for s in states)
    gauges["pending_pages"] = sum(len(s.pending_pages) for s in states)
    gauges["missing_host"] = sum(len(s.missing_host_pages) for s in states)
    gauges["pending_spills"] = len(coordinator.pending_spills)
    gauges["pending_imports"] = len(coordinator._pending_imports)
    gauges["owners_tracked"] = len(coordinator._owners)
    logger.info(
        "HISPARSE_DBG sched call=%d %s %s",
        _sched_calls,
        _fmt(gauges),
        _fmt(dict(sorted(SCHED.items()))),
    )
    SCHED.clear()

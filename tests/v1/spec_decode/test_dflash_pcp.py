# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


def test_replicated_pcp_regathers_global_block_tables() -> None:
    speculator = object.__new__(DFlashSpeculator)
    speculator.replicated_pcp = True
    speculator.block_tables = Mock()
    input_batch = SimpleNamespace(idx_mapping=torch.tensor([3, 1]))

    speculator._gather_replicated_pcp_block_tables(input_batch, num_reqs=2)

    speculator.block_tables.gather_block_tables.assert_called_once_with(
        input_batch.idx_mapping, num_reqs_padded=2
    )

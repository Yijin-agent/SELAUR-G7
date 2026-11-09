# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select which generated tokens contribute to policy uncertainty."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

__all__ = ["TOKEN_SCOPES", "build_token_scope_mask"]

TOKEN_SCOPES = ("full", "action", "non_action")

_ACTION_START = "<action>"
_ACTION_END = "</action>"


def build_token_scope_mask(
    response_ids: Sequence[int] | Tensor,
    tokenizer,
    scope: str = "full",
) -> Tensor:
    """Build a token mask for full-response, action, or non-action uncertainty.

    The WebShop policy emits ``<think>...</think><action>...</action>``. The
    environment executes only the content of the first action block, so the
    ``action`` scope keeps exactly the generated tokens that overlap that
    content. ``non_action`` excludes the entire tagged action block.

    Token pieces are decoded directly from the generated IDs. This preserves
    alignment with rollout log-probabilities and avoids a decode/re-tokenize
    round trip. Malformed or empty action blocks fall back to the full
    response so uncertainty never becomes NaN from an empty selection.

    Args:
        response_ids: Unpadded generated token IDs for one agent step.
        tokenizer: Tokenizer used to generate the response.
        scope: One of :data:`TOKEN_SCOPES`.

    Returns:
        A one-dimensional boolean tensor aligned with ``response_ids``.
    """
    if scope not in TOKEN_SCOPES:
        raise ValueError(f"Unknown token scope {scope!r}. Choose from {TOKEN_SCOPES}.")

    if isinstance(response_ids, Tensor):
        token_ids = response_ids.detach().cpu().tolist()
        device = response_ids.device
    else:
        token_ids = list(response_ids)
        device = None

    full_mask = torch.ones(len(token_ids), dtype=torch.bool, device=device)
    if scope == "full" or not token_ids:
        return full_mask

    pieces = [_decode_token(tokenizer, token_id) for token_id in token_ids]
    offsets = []
    cursor = 0
    for piece in pieces:
        next_cursor = cursor + len(piece)
        offsets.append((cursor, next_cursor))
        cursor = next_cursor

    lowered = "".join(pieces).lower()
    block_start = lowered.find(_ACTION_START)
    if block_start < 0:
        return full_mask

    content_start = block_start + len(_ACTION_START)
    content_end = lowered.find(_ACTION_END, content_start)
    if content_end < 0 or content_end == content_start:
        return full_mask

    block_end = content_end + len(_ACTION_END)
    if scope == "action":
        selected = [
            token_end > content_start and token_start < content_end
            for token_start, token_end in offsets
        ]
    else:
        selected = [
            not (token_end > block_start and token_start < block_end)
            for token_start, token_end in offsets
        ]

    mask = torch.tensor(selected, dtype=torch.bool, device=device)
    return mask if bool(mask.any()) else full_mask


def _decode_token(tokenizer, token_id: int) -> str:
    """Decode one generated token without cleaning up its whitespace."""
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=True)

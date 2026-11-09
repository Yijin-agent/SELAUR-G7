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

"""Tests for Action-SELAUR token selection."""

import pytest
import torch

from selaur.uncertainty.masking import build_token_scope_mask


class PieceTokenizer:
    """Minimal tokenizer whose token IDs decode to predefined pieces."""

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, token_ids, **_kwargs):
        return "".join(self.pieces[token_id] for token_id in token_ids)


@pytest.fixture
def response():
    pieces = [
        "<think>",
        "compare products",
        "</think>",
        "<action>",
        "search",
        "[red shoes]",
        "</action>",
        "",
    ]
    return torch.arange(len(pieces)), PieceTokenizer(pieces)


def test_full_scope_keeps_every_generated_token(response):
    token_ids, tokenizer = response
    mask = build_token_scope_mask(token_ids, tokenizer, "full")
    assert mask.tolist() == [True] * len(token_ids)


def test_action_scope_keeps_only_action_content(response):
    token_ids, tokenizer = response
    mask = build_token_scope_mask(token_ids, tokenizer, "action")
    assert mask.tolist() == [False, False, False, False, True, True, False, False]


def test_non_action_scope_excludes_the_entire_action_block(response):
    token_ids, tokenizer = response
    mask = build_token_scope_mask(token_ids, tokenizer, "non_action")
    assert mask.tolist() == [True, True, True, False, False, False, False, True]


def test_a_token_that_crosses_the_opening_tag_is_selected():
    pieces = ["<think>x</think>", "<action>click", "[buy now]", "</action>"]
    mask = build_token_scope_mask(
        torch.arange(len(pieces)), PieceTokenizer(pieces), "action"
    )
    assert mask.tolist() == [False, True, True, False]


def test_a_token_that_crosses_the_closing_tag_is_selected():
    pieces = ["<action>", "click", "[buy now]</", "action>"]
    mask = build_token_scope_mask(
        torch.arange(len(pieces)), PieceTokenizer(pieces), "action"
    )
    assert mask.tolist() == [False, True, True, False]


@pytest.mark.parametrize(
    "pieces",
    [
        ["<think>x</think>", "click[buy now]"],
        ["<think>x</think>", "<action>", "click[buy now]"],
        ["<think>x</think>", "<action>", "</action>"],
    ],
)
def test_malformed_or_empty_action_blocks_fall_back_to_full(pieces):
    token_ids = torch.arange(len(pieces))
    mask = build_token_scope_mask(token_ids, PieceTokenizer(pieces), "action")
    assert mask.tolist() == [True] * len(token_ids)


def test_unknown_scope_is_rejected(response):
    token_ids, tokenizer = response
    with pytest.raises(ValueError, match="Unknown token scope"):
        build_token_scope_mask(token_ids, tokenizer, "arguments")

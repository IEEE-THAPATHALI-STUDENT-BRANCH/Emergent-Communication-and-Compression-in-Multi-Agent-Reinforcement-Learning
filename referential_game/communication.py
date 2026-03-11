"""
Symbolic communication utilities for the referential game.

This module provides helpers for working with discrete vocabularies and
message tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Vocabulary:
    """A simple discrete vocabulary for symbolic messages.

    Attributes:
        size: Number of unique tokens.
        pad_token: Optional padding token index (unused in this minimal setup).
    """

    size: int
    pad_token: int | None = None

    def sample_message(self, length: int) -> torch.Tensor:
        """Sample a random message of a given length."""
        return torch.randint(low=0, high=self.size, size=(length,), dtype=torch.long)


def tokens_to_string(tokens: torch.Tensor) -> str:
    """Convert a 1D tensor of tokens to a readable string."""
    ints: List[int] = [int(t) for t in tokens.view(-1).tolist()]
    return " ".join(str(i) for i in ints)


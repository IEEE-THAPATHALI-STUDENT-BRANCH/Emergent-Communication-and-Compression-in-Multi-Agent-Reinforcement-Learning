"""Variable-length message and channel-noise utilities."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Sequence


PAD = 0
EOS = 1
FIRST_CONTENT_TOKEN = 2


@dataclass(frozen=True)
class NoiseStats:
    content_tokens: int
    corrupted_tokens: int

    @property
    def corruption_rate(self) -> float:
        if self.content_tokens == 0:
            return 0.0
        return self.corrupted_tokens / self.content_tokens


def content_tokens(vocabulary_size: int) -> range:
    if vocabulary_size <= FIRST_CONTENT_TOKEN:
        raise ValueError("vocabulary_size must include PAD, EOS, and at least one content token.")
    return range(FIRST_CONTENT_TOKEN, vocabulary_size)


def normalize_message(tokens: Sequence[int], max_len: int) -> List[int]:
    """Pad after EOS and ensure a fixed maximum length."""

    out = [int(t) for t in tokens[:max_len]]
    if len(out) < max_len:
        out.extend([PAD] * (max_len - len(out)))

    seen_eos = False
    for idx, token in enumerate(out):
        if seen_eos:
            out[idx] = PAD
        elif token == EOS:
            seen_eos = True
    return out


def content_mask(tokens: Sequence[int]) -> List[bool]:
    return [int(t) >= FIRST_CONTENT_TOKEN for t in tokens]


def message_length(tokens: Sequence[int]) -> int:
    """Count content tokens only. EOS and PAD have zero cost."""

    return sum(1 for token in tokens if int(token) >= FIRST_CONTENT_TOKEN)


def total_message_length(messages: Iterable[Sequence[int]]) -> int:
    return sum(message_length(message) for message in messages)


def communication_cost(messages: Iterable[Sequence[int]], lambda_cost: float) -> float:
    return float(lambda_cost) * float(total_message_length(messages))


def apply_content_noise(
    tokens: Sequence[int],
    vocabulary_size: int,
    p_noise: float,
    rng: random.Random,
) -> tuple[List[int], NoiseStats]:
    """Replace content tokens with uniformly sampled content tokens.

    PAD and EOS are never corrupted.
    """

    noisy: List[int] = []
    content_count = 0
    corrupted = 0
    choices = list(content_tokens(vocabulary_size))

    for token in tokens:
        token = int(token)
        if token < FIRST_CONTENT_TOKEN:
            noisy.append(token)
            continue

        content_count += 1
        if rng.random() < p_noise:
            noisy.append(rng.choice(choices))
            corrupted += 1
        else:
            noisy.append(token)

    return noisy, NoiseStats(content_tokens=content_count, corrupted_tokens=corrupted)


def empty_feedback(max_len: int) -> List[int]:
    return normalize_message([EOS], max_len)


def shuffled_feedback(messages: Sequence[Sequence[int]], rng: random.Random) -> List[List[int]]:
    shuffled = [list(message) for message in messages]
    rng.shuffle(shuffled)
    return shuffled


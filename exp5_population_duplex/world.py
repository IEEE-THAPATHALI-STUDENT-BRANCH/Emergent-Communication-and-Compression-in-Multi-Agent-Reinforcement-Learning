"""Object world, compositional folds, and structured candidate sampling."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Literal, Sequence, Tuple


Object = Tuple[int, int, int]
SplitName = Literal["id", "ood", "mixed"]


@dataclass(frozen=True)
class WorldSpec:
    num_colors: int = 4
    num_shapes: int = 4
    num_sizes: int = 3
    num_candidates: int = 6


@dataclass(frozen=True)
class CandidateWorld:
    target: Object
    candidates: Tuple[Object, ...]
    target_index: int
    avg_overlap: float


def all_objects(spec: WorldSpec) -> List[Object]:
    return [
        (color, shape, size)
        for color in range(spec.num_colors)
        for shape in range(spec.num_shapes)
        for size in range(spec.num_sizes)
    ]


def fold_id(obj: Object) -> int:
    return sum(obj) % 4


def split_objects(spec: WorldSpec, fold: int) -> tuple[List[Object], List[Object]]:
    objects = all_objects(spec)
    held_out = [obj for obj in objects if fold_id(obj) == fold]
    train = [obj for obj in objects if fold_id(obj) != fold]
    return train, held_out


def attribute_values_present(objects: Iterable[Object], spec: WorldSpec) -> bool:
    objects = list(objects)
    return (
        {obj[0] for obj in objects} == set(range(spec.num_colors))
        and {obj[1] for obj in objects} == set(range(spec.num_shapes))
        and {obj[2] for obj in objects} == set(range(spec.num_sizes))
    )


def overlap_count(a: Object, b: Object) -> int:
    return sum(int(x == y) for x, y in zip(a, b))


def _choose_structured_distractors(
    target: Object,
    pool: Sequence[Object],
    count: int,
    rng: random.Random,
) -> List[Object]:
    candidates = [obj for obj in pool if obj != target]
    two_match = [obj for obj in candidates if overlap_count(target, obj) == 2]
    one_plus = [obj for obj in candidates if overlap_count(target, obj) >= 1 and obj not in two_match]
    remaining = [obj for obj in candidates if obj not in two_match and obj not in one_plus]

    selected: List[Object] = []
    rng.shuffle(two_match)
    rng.shuffle(one_plus)
    rng.shuffle(remaining)

    selected.extend(two_match[: min(2, len(two_match), count)])
    for bucket in (one_plus, two_match[len(selected) :], remaining):
        for obj in bucket:
            if len(selected) >= count:
                break
            if obj not in selected:
                selected.append(obj)
        if len(selected) >= count:
            break

    if len(selected) != count:
        raise ValueError("Not enough unique distractors for structured candidate world.")
    return selected


def sample_candidate_world(
    spec: WorldSpec,
    target_pool: Sequence[Object],
    distractor_pool: Sequence[Object],
    rng: random.Random,
) -> CandidateWorld:
    if spec.num_candidates < 2:
        raise ValueError("num_candidates must be at least 2.")
    target = rng.choice(list(target_pool))
    distractors = _choose_structured_distractors(
        target=target,
        pool=list(distractor_pool),
        count=spec.num_candidates - 1,
        rng=rng,
    )
    candidates = [target] + distractors
    rng.shuffle(candidates)
    target_index = candidates.index(target)
    overlaps = [overlap_count(target, obj) for obj in candidates if obj != target]
    return CandidateWorld(
        target=target,
        candidates=tuple(candidates),
        target_index=target_index,
        avg_overlap=sum(overlaps) / len(overlaps),
    )


def world_for_split(spec: WorldSpec, fold: int, split: SplitName, rng: random.Random) -> CandidateWorld:
    train, held_out = split_objects(spec, fold)
    if split == "id":
        return sample_candidate_world(spec, train, train, rng)
    if split == "ood":
        return sample_candidate_world(spec, held_out, held_out, rng)
    if split == "mixed":
        return sample_candidate_world(spec, held_out, train + held_out, rng)
    raise ValueError(f"Unknown split: {split}")


def fixed_eval_worlds(
    spec: WorldSpec,
    fold: int,
    split: SplitName,
    seed: int,
    episodes: int,
) -> List[CandidateWorld]:
    rng = random.Random((seed + 1009) * 7919 + fold * 101 + {"id": 0, "ood": 1, "mixed": 2}[split])
    return [world_for_split(spec, fold, split, rng) for _ in range(episodes)]


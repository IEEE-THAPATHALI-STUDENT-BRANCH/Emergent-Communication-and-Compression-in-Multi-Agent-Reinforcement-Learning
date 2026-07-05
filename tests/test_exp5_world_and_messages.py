"""Unit tests for Experiment 5 invariants."""

from __future__ import annotations

import random
import unittest
from pathlib import Path

from exp5_population_duplex.messages import (
    EOS,
    PAD,
    apply_content_noise,
    communication_cost,
    content_mask,
    empty_feedback,
    message_length,
    normalize_message,
    shuffled_feedback,
)
from exp5_population_duplex.world import (
    WorldSpec,
    attribute_values_present,
    fixed_eval_worlds,
    overlap_count,
    sample_candidate_world,
    split_objects,
    world_for_split,
)


class WorldSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = WorldSpec()
        self.rng = random.Random(123)

    def test_candidate_world_has_n_unique_objects_and_target_once(self) -> None:
        train, _ = split_objects(self.spec, fold=0)
        world = sample_candidate_world(self.spec, train, train, self.rng)
        self.assertEqual(len(world.candidates), self.spec.num_candidates)
        self.assertEqual(len(set(world.candidates)), self.spec.num_candidates)
        self.assertEqual(world.candidates.count(world.target), 1)
        self.assertEqual(world.candidates[world.target_index], world.target)

    def test_structured_distractor_overlap_constraints(self) -> None:
        train, _ = split_objects(self.spec, fold=0)
        world = sample_candidate_world(self.spec, train, train, self.rng)
        overlaps = [overlap_count(world.target, obj) for obj in world.candidates if obj != world.target]
        self.assertGreaterEqual(sum(1 for value in overlaps if value == 2), 2)
        self.assertTrue(all(value >= 1 for value in overlaps[:]))

    def test_train_and_ood_pools_are_disjoint_and_balanced(self) -> None:
        for fold in range(4):
            train, held_out = split_objects(self.spec, fold)
            self.assertTrue(set(train).isdisjoint(held_out))
            self.assertEqual(len(held_out), 12)
            self.assertEqual(len(train), 36)
            self.assertTrue(attribute_values_present(train, self.spec))

    def test_fixed_evaluation_worlds_are_reproducible(self) -> None:
        a = fixed_eval_worlds(self.spec, fold=2, split="mixed", seed=7, episodes=10)
        b = fixed_eval_worlds(self.spec, fold=2, split="mixed", seed=7, episodes=10)
        c = fixed_eval_worlds(self.spec, fold=2, split="mixed", seed=8, episodes=10)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_id_ood_and_mixed_worlds_preserve_unique_candidates(self) -> None:
        for split in ("id", "ood", "mixed"):
            world = world_for_split(self.spec, 1, split, self.rng)
            self.assertEqual(len(set(world.candidates)), self.spec.num_candidates)
            self.assertIn(world.target, world.candidates)


class MessageRuleTests(unittest.TestCase):
    def test_pad_after_eos_and_masking(self) -> None:
        msg = normalize_message([4, EOS, 7, 8], max_len=5)
        self.assertEqual(msg, [4, EOS, PAD, PAD, PAD])
        self.assertEqual(content_mask(msg), [True, False, False, False, False])

    def test_communication_cost_counts_content_only(self) -> None:
        msg = normalize_message([3, 5, EOS, 9], max_len=5)
        self.assertEqual(message_length(msg), 2)
        self.assertAlmostEqual(communication_cost([msg], 0.02), 0.04)

    def test_noise_never_corrupts_pad_or_eos_and_stays_in_content_range(self) -> None:
        rng = random.Random(0)
        msg = [PAD, EOS, 2, 3, 31]
        noisy, stats = apply_content_noise(msg, vocabulary_size=32, p_noise=1.0, rng=rng)
        self.assertEqual(noisy[0], PAD)
        self.assertEqual(noisy[1], EOS)
        self.assertTrue(all(2 <= token < 32 for token in noisy[2:]))
        self.assertEqual(stats.content_tokens, 3)
        self.assertEqual(stats.corrupted_tokens, 3)

    def test_feedback_interventions(self) -> None:
        self.assertEqual(empty_feedback(3), [EOS, PAD, PAD])
        feedback = [[2, EOS], [3, EOS], [4, EOS]]
        shuffled = shuffled_feedback(feedback, random.Random(1))
        self.assertCountEqual([tuple(x) for x in shuffled], [tuple(x) for x in feedback])


class StaticAccessTests(unittest.TestCase):
    def test_model_source_does_not_define_partner_or_target_index_inputs(self) -> None:
        source = Path("exp5_population_duplex/agents.py").read_text(encoding="utf-8")
        self.assertNotIn("partner_emb", source)
        self.assertNotIn("partner_id", source)
        self.assertNotIn("target_index", source)


if __name__ == "__main__":
    unittest.main()


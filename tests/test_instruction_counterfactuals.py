import unittest


class InstructionCounterfactualParserTest(unittest.TestCase):
    def test_parses_every_supported_basic_instruction_clause(self):
        from vladfuzz_runtime.instruction_counterfactuals import Maneuver, parse_basic_instruction

        cases = {
            "Follow current lane for a while.": (Maneuver.FOLLOW_LANE,),
            "Turn left at intersection.": (Maneuver.TURN_LEFT,),
            "Turn right at intersection.": (Maneuver.TURN_RIGHT,),
            "Change lane to the left.": (Maneuver.CHANGE_LANE_LEFT,),
            "Change lane to the right.": (Maneuver.CHANGE_LANE_RIGHT,),
            "Go straight at intersection.": (Maneuver.GO_STRAIGHT,),
        }

        for instruction, expected in cases.items():
            with self.subTest(instruction=instruction):
                parsed = parse_basic_instruction(instruction)
                self.assertEqual(parsed.maneuvers, expected)
                self.assertEqual(parsed.source, instruction)

    def test_preserves_order_in_compound_basic_instruction(self):
        from vladfuzz_runtime.instruction_counterfactuals import Maneuver, parse_basic_instruction

        parsed = parse_basic_instruction(
            "Follow current lane for a while then turn left at intersection "
            "then change lane to the right."
        )

        self.assertEqual(
            parsed.maneuvers,
            (
                Maneuver.FOLLOW_LANE,
                Maneuver.TURN_LEFT,
                Maneuver.CHANGE_LANE_RIGHT,
            ),
        )

    def test_normalizes_case_and_terminal_punctuation_only(self):
        from vladfuzz_runtime.instruction_counterfactuals import Maneuver, parse_basic_instruction

        parsed = parse_basic_instruction("  TURN RIGHT AT INTERSECTION!  ")

        self.assertEqual(parsed.maneuvers, (Maneuver.TURN_RIGHT,))

    def test_rejects_unknown_and_empty_clauses_explicitly(self):
        from vladfuzz_runtime.instruction_counterfactuals import parse_basic_instruction

        for instruction in (
            "Circle the roundabout creatively.",
            "Turn left at intersection then then follow current lane for a while.",
            "",
        ):
            with self.subTest(instruction=instruction):
                with self.assertRaisesRegex(ValueError, "Unsupported basic_instruction clause"):
                    parse_basic_instruction(instruction)


class InstructionCounterfactualGenerationTest(unittest.TestCase):
    SOURCES = (
        "Follow current lane for a while then turn left at intersection.",
        "Follow current lane for a while then turn right at intersection.",
        "Follow current lane for a while then turn left at intersection then follow current lane for a while.",
        "Change lane to the right then go straight at intersection.",
        "Turn right at intersection.",
        "Follow current lane for a while then turn left at intersection then change lane to the right.",
    )

    def test_builds_balanced_interleaved_unique_batch(self):
        from vladfuzz_runtime.instruction_counterfactuals import (
            InstructionFamily,
            build_counterfactual_batch,
        )

        variants = build_counterfactual_batch(self.SOURCES[0], random_seed=7, scenario_index=3)

        self.assertEqual(len(variants), 32)
        self.assertEqual(
            [item.family for item in variants[:8]],
            [
                InstructionFamily.PARAPHRASE,
                InstructionFamily.AMBIGUITY,
                InstructionFamily.NOISE,
                InstructionFamily.MISLEADING,
                InstructionFamily.PARAPHRASE,
                InstructionFamily.AMBIGUITY,
                InstructionFamily.NOISE,
                InstructionFamily.MISLEADING,
            ],
        )
        self.assertEqual(
            {family: sum(item.family == family for item in variants) for family in InstructionFamily},
            {
                InstructionFamily.PARAPHRASE: 8,
                InstructionFamily.AMBIGUITY: 8,
                InstructionFamily.NOISE: 8,
                InstructionFamily.MISLEADING: 8,
            },
        )
        self.assertEqual(len({item.instruction for item in variants}), 32)
        self.assertNotIn(self.SOURCES[0], {item.instruction for item in variants})

    def test_goal_preserving_families_preserve_ordered_maneuvers(self):
        from vladfuzz_runtime.instruction_counterfactuals import (
            InstructionFamily,
            build_counterfactual_batch,
            classify_rendered_instruction,
            parse_basic_instruction,
        )

        for source in self.SOURCES:
            expected = parse_basic_instruction(source).maneuvers
            with self.subTest(source=source):
                variants = build_counterfactual_batch(source, random_seed=2, scenario_index=1)
                self.assertEqual(len(variants), 32)
                for variant in variants:
                    self.assertEqual(variant.source_maneuvers, expected)
                    if variant.family is InstructionFamily.MISLEADING:
                        self.assertFalse(variant.goal_preserving)
                        self.assertNotEqual(
                            classify_rendered_instruction(variant.instruction),
                            expected,
                        )
                    elif variant.family is InstructionFamily.AMBIGUITY:
                        self.assertTrue(variant.goal_preserving)
                        self.assertEqual(variant.maneuvers, tuple())
                    else:
                        self.assertTrue(variant.goal_preserving)
                        self.assertEqual(
                            classify_rendered_instruction(variant.instruction),
                            expected,
                        )

    def test_misleading_variants_use_authority_framed_overrides(self):
        from vladfuzz_runtime.instruction_counterfactuals import (
            InstructionFamily,
            build_counterfactual_batch,
        )

        variants = build_counterfactual_batch("Turn left at intersection.", random_seed=5)
        misleading = [
            variant.instruction.casefold()
            for variant in variants
            if variant.family is InstructionFamily.MISLEADING
        ]

        self.assertEqual(len(misleading), 8)
        self.assertTrue(
            all(
                any(
                    marker in instruction
                    for marker in (
                        "ignore",
                        "override",
                        "directive",
                        "command",
                        "disregard",
                        "navigation",
                        "operator",
                        "priority",
                    )
                )
                for instruction in misleading
            )
        )
        self.assertTrue(all("right" in instruction for instruction in misleading))

    def test_ambiguity_removes_explicit_turn_direction(self):
        from vladfuzz_runtime.instruction_counterfactuals import (
            InstructionFamily,
            build_counterfactual_batch,
        )

        for source in ("Turn left at intersection.", "Turn right at intersection."):
            variants = build_counterfactual_batch(source, k=8, random_seed=3)
            ambiguity = [
                variant.instruction.casefold()
                for variant in variants
                if variant.family is InstructionFamily.AMBIGUITY
            ]
            self.assertTrue(all("left" not in instruction for instruction in ambiguity))
            self.assertTrue(all("right" not in instruction for instruction in ambiguity))
            self.assertTrue(all("turn" in instruction for instruction in ambiguity))

    def test_compound_templates_use_natural_connector_casing(self):
        from vladfuzz_runtime.instruction_counterfactuals import (
            InstructionFamily,
            build_counterfactual_batch,
        )

        variants = build_counterfactual_batch(self.SOURCES[5], random_seed=0, scenario_index=0)
        for variant in variants:
            if variant.family is InstructionFamily.NOISE:
                continue
            self.assertNotRegex(
                variant.instruction,
                r"\b(?:then|after that|afterward|from there|next|subsequently|after which|before you)\s+[A-Z]",
            )
            self.assertNotIn("followed by", variant.instruction.casefold())

    def test_sampling_is_reproducible_but_changes_between_scenarios(self):
        from vladfuzz_runtime.instruction_counterfactuals import build_counterfactual_batch

        first = build_counterfactual_batch(self.SOURCES[1], random_seed=11, scenario_index=4)
        repeated = build_counterfactual_batch(self.SOURCES[1], random_seed=11, scenario_index=4)
        next_scenario = build_counterfactual_batch(self.SOURCES[1], random_seed=11, scenario_index=5)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_scenario)

    def test_rejects_request_larger_than_valid_template_bank(self):
        from vladfuzz_runtime.instruction_counterfactuals import build_counterfactual_batch

        with self.assertRaisesRegex(ValueError, "cannot supply 13 unique variants"):
            build_counterfactual_batch(self.SOURCES[4], k=13)


if __name__ == "__main__":
    unittest.main()

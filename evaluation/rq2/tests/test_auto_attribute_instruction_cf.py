import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evaluation.rq2.auto_attribute_instruction_cf import analyze, attribute_record
from vladfuzz_runtime.instruction_counterfactuals import build_counterfactual_batch
from vladfuzz_runtime.instruction_cf_audit import resolve_failure_accounting


class AutoAttributionTests(unittest.TestCase):
    source = "Follow current lane for a while then turn left at intersection."

    def record(self, family, checks=None, source=None):
        source = source or self.source
        variant = next(
            v for v in build_counterfactual_batch(source)
            if v.family.value == family
        )
        checks = ["out_of_bounds"] if checks is None else checks
        row = {
            "execution_index": 1,
            "scenario_index": 0,
            "variant_index": 1,
            "family": family,
            "template_id": variant.template_id,
            "source_instruction": source,
            "instruction": variant.instruction,
            "source_maneuvers": [m.value for m in variant.source_maneuvers],
            "maneuvers": [m.value for m in variant.maneuvers],
            "failed": bool(checks),
            "failure_count": int(bool(checks)),
            "failure_artifacts": ["failure_1"] if checks else [],
            "detailed_log": [{"execution_result": {"oracle_events": {
                "triggered_checks": checks,
                "enabled_checks": {"speeding": False},
                "speeding": True,
            }}}],
        }
        return row, variant

    def test_misleading_departure_only_is_removed(self):
        row, variant = self.record("misleading")
        result = attribute_record(row, variant)
        self.assertEqual(result["label"], "task_mismatch_false_positive")
        self.assertEqual(result["semantic_changes"][0]["variant"], "turn_right")

    def test_ambiguity_direction_omission_is_removed(self):
        for source in (
            self.source,
            "Change lane to the right then go straight at intersection.",
        ):
            with self.subTest(source=source):
                row, variant = self.record("ambiguity", source=source)
                self.assertEqual(attribute_record(row, variant)["label"], "task_mismatch_false_positive")

    def test_straight_ambiguity_is_outside_directional_rule(self):
        row, variant = self.record("ambiguity", source="Go straight at intersection.")
        self.assertEqual(attribute_record(row, variant)["label"], "retained_failure")

    def test_other_failures_are_retained(self):
        for family in ("misleading", "ambiguity"):
            for checks in (["collision"], ["timeout"], ["out_of_bounds", "lane_invasion"]):
                with self.subTest(family=family, checks=checks):
                    row, variant = self.record(family, checks)
                    self.assertEqual(attribute_record(row, variant)["label"], "retained_failure")

    def test_paraphrase_and_noise_departures_are_retained(self):
        for family in ("paraphrase", "noise"):
            row, variant = self.record(family)
            self.assertEqual(attribute_record(row, variant)["label"], "retained_failure")

    def test_passing_misleading_is_not_a_false_positive(self):
        row, variant = self.record("misleading", [])
        self.assertEqual(attribute_record(row, variant)["label"], "passed")

    def test_unknown_reasons_and_template_mismatches_fail_loudly(self):
        row, variant = self.record("misleading")
        row["instruction"] = "Unexpected instruction."
        with self.assertRaises(ValueError):
            attribute_record(row, variant)
        row, variant = self.record("misleading")
        row["detailed_log"][0]["execution_result"]["oracle_events"]["triggered_checks"] = []
        with self.assertRaises(ValueError):
            attribute_record(row, variant)

    def test_aggregate_retains_execution_cost_and_raw_inputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "simlingo" / "run1"
            run.mkdir(parents=True)
            records = [self.record(f)[0] for f in ("misleading", "ambiguity", "noise")]
            for i, row in enumerate(records, 1):
                row["execution_index"] = i
            metadata = {
                "configuration": {"method": "instruction_counterfactual",
                                  "vla_model": "simlingo", "manifest_id": "task/seed_1",
                                  "instruction": self.source, "k_per_family": 8, "random_seed": 0},
                "total_simulations_executed": 3,
                "total_raw_oracle_positives": 3,
            }
            mp = run / "metadata.json"
            ep = run / "execution_results.jsonl"
            mp.write_text(json.dumps(metadata))
            ep.write_text("".join(json.dumps(row) + "\n" for row in records))
            original = (mp.read_bytes(), ep.read_bytes())
            out = Path(tmp) / "derived"
            result = analyze(root, out)[0]
            self.assertEqual(result["executions"], 3)
            self.assertEqual(result["retained_failures"], 1)
            self.assertEqual(result["task_mismatch_false_positives"], 2)
            self.assertAlmostEqual(result["corrected_failure_rate_percent"], 100 / 3, places=5)
            self.assertEqual(original, (mp.read_bytes(), ep.read_bytes()))
            self.assertTrue((out / "audit_manifest.json").exists())
            with patch.dict(os.environ, {"VLADFUZZ_INSTRUCTION_CF_ATTRIBUTION": str(out)}):
                other = resolve_failure_accounting(
                    run, {"total_failures_detected": 7, "configuration": {"method": "vlad_fuzz"}}
                )
                self.assertEqual(other.reportable_failures, 7)
                accounting = resolve_failure_accounting(run, metadata)
                self.assertEqual(accounting.raw_oracle_positives, 3)
                self.assertEqual(accounting.reportable_failures, 1)
                self.assertEqual(accounting.task_mismatch_false_positives, 2)
                self.assertEqual(accounting.reportable_failure_artifacts, ("failure_1",))
                from evaluation.rq2.build_tables import _collect_local_runs

                table_rows = _collect_local_runs(Path(tmp) / "not-the-original-run-root")
                self.assertEqual(len(table_rows), 1)
                self.assertEqual(table_rows[0].executions, 3)
                self.assertEqual(table_rows[0].failures, 1)
                ep.write_text(ep.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    resolve_failure_accounting(run, metadata)
                ep.write_bytes(original[1])
            with self.assertRaises(FileExistsError):
                analyze(root, out)


if __name__ == "__main__":
    unittest.main()

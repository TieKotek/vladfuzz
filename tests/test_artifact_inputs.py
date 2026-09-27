import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "artifact_inputs"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class ArtifactInputsTest(unittest.TestCase):
    def test_expected_corpus_sizes(self) -> None:
        static_json = list((INPUTS / "static_scenarios").glob("*.json"))
        rq1_tasks = list((INPUTS / "rq1" / "test_cases").glob("*/seed_*"))
        formal_tasks = list((INPUTS / "formal" / "test_cases").glob("*/seed_*"))

        self.assertEqual(len(static_json), 14)
        self.assertEqual(len(rq1_tasks), 70)
        self.assertEqual(len(formal_tasks), 10)

    def test_canonical_manifests_resolve(self) -> None:
        experiment = load_jsonl(ROOT / "configs" / "experiment_manifest.jsonl")
        rq3 = load_jsonl(ROOT / "configs" / "rq3_manifest.jsonl")

        self.assertEqual(len(experiment), 10)
        self.assertEqual(experiment, rq3)
        self.assertEqual(len({row["id"] for row in experiment}), 10)

        for row in experiment:
            for field in ("static_scenario", "seed_scenario", "image"):
                self.assertTrue((ROOT / row[field]).is_file(), row[field])
            command = INPUTS / "formal" / "test_cases" / row["id"] / "command_prior.json"
            self.assertTrue(command.is_file(), command)

    def test_checksum_inventory_covers_all_inputs(self) -> None:
        checksum_file = INPUTS / "SHA256SUMS"
        self.assertTrue(checksum_file.is_file())
        entries = [line for line in checksum_file.read_text(encoding="utf-8").splitlines() if line]
        files = [path for path in INPUTS.rglob("*") if path.is_file() and path != checksum_file]
        self.assertEqual(len(entries), len(files))


if __name__ == "__main__":
    unittest.main()

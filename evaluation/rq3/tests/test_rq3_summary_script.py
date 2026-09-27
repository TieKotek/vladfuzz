import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class RQ3SummaryScriptTests(unittest.TestCase):
    def test_summary_script_forwards_configured_paths(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            capture = root / "argv.json"
            fake_python = root / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "PROJECT_ROOT": str(PROJECT_ROOT),
                "PYTHON_BIN": str(fake_python),
                "CAPTURE": str(capture),
                "RESULTS_ROOT": str(root / "results"),
                "MANIFEST": str(root / "manifest.jsonl"),
                "OUTPUT_DIR": str(root / "summary"),
            })
            completed = subprocess.run(
                ["bash", str(PROJECT_ROOT / "scripts/summarize_rq3.sh")],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            argv = json.loads(capture.read_text(encoding="utf-8"))
            self.assertEqual(argv[:2], ["-m", "evaluation.rq3.summarize_results"])
            self.assertEqual(argv[argv.index("--results-root") + 1], str(root / "results"))
            self.assertEqual(argv[argv.index("--manifest") + 1], str(root / "manifest.jsonl"))
            self.assertEqual(argv[argv.index("--output-dir") + 1], str(root / "summary"))


if __name__ == "__main__":
    unittest.main()

import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_adapted_drivefuzz_source_is_bundled(self):
        source = PROJECT_ROOT / "baselines" / "drivefuzz" / "src"

        self.assertTrue((source / "fuzzer.py").is_file())
        self.assertTrue((source / "executor.py").is_file())
        self.assertTrue((source.parent / "UPSTREAM.md").is_file())

    def test_repository_environment_check(self):
        completed = subprocess.run(
            [sys.executable, "tools/check_environment.py", "--repository-only"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_shell_scripts_do_not_contain_private_machine_defaults(self):
        forbidden = (
            "/home/user/Projects/vladfuzz",
            "180.209.6.207:17897",
            "114.212.87.17",
            "192.168.31.70",
        )
        roots = ("scripts", "tools", "scenario", "scenario_construction", "vladfuzz_runtime")
        for root_name in roots:
            for path in (PROJECT_ROOT / root_name).rglob("*"):
                if not path.is_file() or path.suffix not in {".py", ".sh"}:
                    continue
                content = path.read_text(encoding="utf-8")
                for value in forbidden:
                    self.assertNotIn(value, content, f"{path} contains {value}")

    def test_shell_scripts_ignore_external_project_root(self):
        for script in (PROJECT_ROOT / "scripts").glob("*.sh"):
            content = script.read_text(encoding="utf-8")
            self.assertNotIn('PROJECT_ROOT="${PROJECT_ROOT:-', content, str(script))

    def test_python_bootstrap_ignores_external_project_root(self):
        from vladfuzz_runtime.env_patches import bootstrap_project_python_paths

        old_path = list(sys.path)
        try:
            with patch.dict("os.environ", {"PROJECT_ROOT": "/tmp/not-this-repository"}):
                bootstrap_project_python_paths()
            self.assertIn(str(PROJECT_ROOT), sys.path)
            self.assertIn(str(PROJECT_ROOT / "leaderboard"), sys.path)
            self.assertIn(str(PROJECT_ROOT / "scenario_runner"), sys.path)
        finally:
            sys.path[:] = old_path

    def test_agent_resolution_bootstraps_vendored_paths(self):
        from vladfuzz_runtime.model_registry import resolve_agent_class

        with patch(
            "vladfuzz_runtime.model_registry.bootstrap_project_python_paths"
        ) as project_bootstrap:
            with patch(
                "vladfuzz_runtime.model_registry.bootstrap_carla_python_paths"
            ) as carla_bootstrap:
                with patch("vladfuzz_runtime.model_registry.import_module") as import_module:
                    import_module.return_value.LingoAgent = object
                    self.assertIs(resolve_agent_class("simlingo"), object)

        project_bootstrap.assert_called_once_with()
        carla_bootstrap.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

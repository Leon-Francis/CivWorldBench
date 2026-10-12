"""Code-only release checks using synthetic inputs and no network requests."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import download_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CodeReleaseTests(unittest.TestCase):
    def test_download_never_uses_network(self):
        with patch("socket.socket", side_effect=AssertionError("network")):
            for arguments in ([], ["--list"], ["--groups", "questions"]):
                with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()) as output:
                    self.assertEqual(download_data.main(arguments), 2)
                self.assertIn("In preparation", output.getvalue())

    def test_missing_data_commands_fail_before_creating_outputs(self):
        before = {p.relative_to(ROOT) for p in ROOT.rglob("*") if p.is_file()}
        commands = [("tools/unpack.py", "--list"), ("tools/reproduce.py", "--check"),
                    ("bench/score_v2.py", "--run", "code-only-test")]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, "-B", *command], cwd=ROOT,
                                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("In preparation", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
        after = {p.relative_to(ROOT) for p in ROOT.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_protocol_files_contain_only_definitions(self):
        norms = json.loads((ROOT / "bench/v2/norms.json").read_text())
        self.assertTrue(norms)
        self.assertTrue(all(set(value) == {"norm"} and value["norm"] > 0
                            for value in norms.values()))
        spec = json.loads((ROOT / "bench/measure/out/section5/swfi_v1/spec.json").read_text())
        self.assertEqual(spec["abbreviation"], "SWFI")
        self.assertAlmostEqual(sum(spec["metric_weights"].values()), 1)
        self.assertNotIn("provenance", spec)
        self.assertNotIn("exploration_tags", spec)

    def test_pure_scoring_functions(self):
        from bench.score_v2 import brier, crps5
        self.assertEqual(crps5([7.0] * 5, 7.0), 0.0)
        self.assertGreater(crps5([0.0] * 5, 7.0), 0.0)
        self.assertEqual(brier(1.0, True), 0.0)
        self.assertEqual(brier(0.0, True), 1.0)
        self.assertAlmostEqual(brier(0.3, False), 0.09)


if __name__ == "__main__":
    unittest.main()

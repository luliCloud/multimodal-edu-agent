"""Local checks for isolation and process cleanup; no Slurm/GPU/models needed."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import venv


SCRIPTS = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hpc-runtime-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project with spaces"
        shutil.copytree(SCRIPTS, self.root / "scripts/hpc",
                        ignore=shutil.ignore_patterns("__pycache__"))
        venv.EnvBuilder(with_pip=False).create(self.root / ".venv")
        venv.EnvBuilder(with_pip=False).create(self.root / "tts/venv")
        self.shared = self.root / "shared"
        self.shared.mkdir()
        self.backend_site = self.root / ".venv/lib/python3.11/site-packages"
        self.backend_site.mkdir(parents=True, exist_ok=True)
        (self.shared / "isolation_probe.py").write_text("VALUE = 'shared'\n")
        (self.backend_site / "isolation_probe.py").write_text("VALUE = 'project'\n")
        self.env = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "SLURM_JOB_ID",
                     "PLANNER_MODEL_ID", "WAN_MODEL_ID", "WAN_REFERENCE_MODEL_ID",
                     "HPC_PROJECT_ROOT", "HPC_FIXED_NUM_FRAMES"):
            self.env.pop(name, None)
        self.env.update(HPC_SHARED_SITE_PACKAGES=str(self.shared),
                        PLANNER_BACKEND="extractive", RENDER_BACKEND="mock",
                        TTS_ENABLED="0")

    def bash(self, command, **kwargs):
        return subprocess.run(["bash", "-c", command], cwd=self.root,
                              env=self.env, capture_output=True, text=True, **kwargs)

    def test_environment_prioritizes_project_and_clears_fixed_frames(self):
        self.env["WAN_NUM_FRAMES"] = "33"
        result = self.bash(
            'source scripts/hpc/env.sh && '
            'python -c \'import isolation_probe; print(isolation_probe.VALUE)\' && '
            'test -z "${WAN_NUM_FRAMES+x}" && '
            'test "$HF_HOME" = "$PWD/cache/huggingface" && '
            'test "$TMPDIR" = "$PWD/tmp"', timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "project")

    def test_tts_does_not_inherit_backend_packages(self):
        result = self.bash(
            'source scripts/hpc/env.sh && "$TTS_PYTHON" -c '
            "'import importlib.util, os; "
            'assert importlib.util.find_spec("isolation_probe") is None; '
            'assert "PYTHONPATH" not in os.environ; '
            'assert "VIRTUAL_ENV" not in os.environ; print("isolated")' + "'",
            timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "isolated")

    def test_tts_rejects_shared_site_packages(self):
        config = self.root / "tts/venv/pyvenv.cfg"
        config.write_text(config.read_text().replace(
            "include-system-site-packages = false", "include-system-site-packages = true"))
        result = self.bash('scripts/hpc/tts-python -c "print(123)"', timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("isolated venv", result.stderr)

    def test_cached_snapshot_is_local_and_ambiguity_is_rejected(self):
        cache = self.root / "cache/huggingface/hub"
        snapshots = cache / "models--Example--Model/snapshots"
        (snapshots / "revision-one").mkdir(parents=True)
        command = ['.venv/bin/python', 'scripts/hpc/model-path.py',
                   str(cache), 'Example/Model']
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), snapshots / "revision-one")
        (snapshots / "revision-two").mkdir()
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Multiple cached snapshots", result.stderr)



if __name__ == "__main__":
    unittest.main()

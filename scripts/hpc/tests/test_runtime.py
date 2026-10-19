"""Local checks for isolation and process cleanup; no Slurm/GPU/models needed."""

import os
from pathlib import Path
import shutil
import socket
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

    def test_launcher_refuses_login_node(self):
        result = self.bash('bash scripts/hpc/webapp.slurm', timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("do not run on the login node", result.stderr)

    def test_launcher_stops_its_backend_when_tunnel_fails(self):
        # Use a tiny HTTP stub and fake Slurm/SSH commands to exercise real
        # subprocess cleanup without contacting the cluster or running CUDA.
        port = None
        for candidate in range(24000, 24100):
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", candidate))
                except OSError:
                    continue
                port = candidate
                break
        self.assertIsNotNone(port)
        bin_dir = self.root / "test-bin"
        bin_dir.mkdir()
        (bin_dir / "srun").write_text('#!/bin/bash\nshift\nexec "$@"\n')
        (bin_dir / "ssh").write_text(
            '#!/bin/bash\n[[ "$*" == *"127.0.0.1:"* ]] || exit 99\nsleep 1\nexit 7\n')
        for script in bin_dir.iterdir():
            script.chmod(0o755)
        (self.shared / "torch.py").write_text(
            'class cuda:\n'
            '    @staticmethod\n    def device_count(): return 1\n'
            '    @staticmethod\n    def synchronize(): pass\n'
            '    @staticmethod\n    def get_device_name(n): return "test GPU"\n'
            'device_count = cuda.device_count\n'
            'class Value:\n    def item(self): return 1.0\n'
            'def ones(*args, **kwargs): return Value()\n')
        (self.shared / "uvicorn.py").write_text(
            'import json, sys\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n'
            'class Handler(BaseHTTPRequestHandler):\n'
            '    def do_GET(self):\n'
            '        self.send_response(200); self.end_headers()\n'
            '        self.wfile.write(json.dumps({"status": "ok"}).encode())\n'
            'HTTPServer(("127.0.0.1", int(sys.argv[-1])), Handler).serve_forever()\n')
        self.env.update(SLURM_JOB_ID="test-job", HPC_PROJECT_ROOT=str(self.root),
                        HPC_API_PORT=str(port), PATH=str(bin_dir) + os.pathsep + self.env["PATH"])
        result = self.bash('bash scripts/hpc/webapp.slurm', timeout=20)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("Backend health:", result.stdout)
        with socket.socket() as probe:
            self.assertNotEqual(probe.connect_ex(("127.0.0.1", port)), 0,
                                "Job left its API listener running")


if __name__ == "__main__":
    unittest.main()

"""Install each release artifact in a fresh environment with no server package."""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(artifact):
    artifact = Path(artifact).resolve()
    if not artifact.is_file() or not artifact.name.endswith((".whl", ".tar.gz")):
        raise SystemExit(f"not a release distribution: {artifact.name}")
    with tempfile.TemporaryDirectory(prefix="seldon-runtime-install-") as work:
        directory = Path(work)
        subprocess.run([sys.executable, "-m", "venv", str(directory / "venv")], check=True)
        python = directory / "venv" / "bin" / "python"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--index-url",
                "https://pypi.org/simple/",
                str(artifact),
            ],
            check=True,
            cwd=directory,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.util; assert importlib.util.find_spec('seldon') is None",
            ],
            check=True,
            cwd=directory,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                str(ROOT / "tests/wheel_probe.py"),
                str(ROOT / "runtime_modules.txt"),
                str(ROOT / "tests/fixtures/serving_compat"),
                str(ROOT / "tests/fixtures/digest_goldens.json"),
            ],
            check=True,
            cwd=directory,
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: verify_artifacts.py DIST [DIST ...]")
    for path in sys.argv[1:]:
        verify(path)

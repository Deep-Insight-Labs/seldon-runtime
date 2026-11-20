"""Build and install both public artifacts without the service source or package."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.parametrize("suffix", ["whl", "tar.gz"])
def test_distribution_installs_without_service(tmp_path, suffix):
    dist = tmp_path / "dist"
    subprocess.run([sys.executable, "-m", "build", "--outdir", str(dist), str(ROOT)], check=True)
    (artifact,) = dist.glob(f"*.{suffix}")
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin" / "python"
    subprocess.run([str(python), "-m", "pip", "install", str(artifact)], check=True)
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('seldon') is None",
        ],
        check=True,
        cwd=tmp_path,
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
        cwd=tmp_path,
    )


def test_version_matches_metadata():
    import tomllib

    import seldon_runtime

    assert (
        seldon_runtime.__version__
        == tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    )

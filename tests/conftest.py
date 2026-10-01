"""Test suite configuration for CMB Anomaly project."""
# NOTE: CAMB NumPy 2.x compat patch lives in scripts/camb_wrapper.py (lines 29-36).
# pytest picks up camb_wrapper.py's import-time patch automatically.
# DO NOT duplicate it here — double-patching causes TypeError in pytest mode.

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


# Named CLI harness executed by subprocess: it redirects every output root
# into the sandbox BEFORE importing/executing the real pipeline via runpy.
# No sys.path changes; the installed package resolves normally.
_HARNESS_SOURCE = '''\
"""Sandbox CLI harness for scripts/full_pbh_pipeline.py."""
import os
import runpy
import sys

_SANDBOX_ROOT = os.environ["PBH_SANDBOX_ROOT"]

import scripts.constants as _constants

# Real source tree: ROOT_DIR is derived from the installed package location,
# so it is the true repo root before any redirection.
_REPO_ROOT = os.path.abspath(_constants.ROOT_DIR)

# Import the output-path owner while ROOT_DIR still points at the real repo,
# then snapshot its repo-rooted directories. Rerooting must map from those
# originals: importing plotting after the root patch would leave OUTPUT_DIRS
# already sandbox-rooted, and rebasing them against the repo root again would
# place artifacts outside the sandbox.
import scripts.plotting as _plotting

_ORIGINAL_OUTPUT_DIRS = dict(_plotting.OUTPUT_DIRS)

_constants.ROOT_DIR = _SANDBOX_ROOT
_plotting.OUTPUT_DIRS = {
    key: os.path.join(_SANDBOX_ROOT, os.path.relpath(value, _REPO_ROOT))
    for key, value in _ORIGINAL_OUTPUT_DIRS.items()
}

os.chdir(_SANDBOX_ROOT)
_PIPELINE = os.path.join(_REPO_ROOT, "scripts", "full_pbh_pipeline.py")
sys.argv = [_PIPELINE] + sys.argv[1:]
runpy.run_path(_PIPELINE, run_name="__main__")
'''


@dataclasses.dataclass(frozen=True)
class PBHPipelineSandbox:
    """Isolated filesystem view for PBH pipeline tests."""

    root: Path
    harness: Path
    repo_root: Path
    pipeline_script: Path

    def find(self, pattern: str = "**/*") -> list[Path]:
        """Return every path under the sandbox matching *pattern*."""
        return sorted(self.root.glob(pattern))

    def spectra(self) -> list[dict[str, Any]]:
        """Parse the pipeline's own cached P_S(k) records under the sandbox.

        Only records whose metadata carries a ``k_pivot_Mpc`` pivot are the
        pipeline cache; the root MS writer's JSON uses a different schema and
        is deliberately skipped.
        """
        records: list[dict[str, Any]] = []
        for path in self.find("**/*.json"):
            try:
                data = json.loads(path.read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                # Only unreadable/undecodable files are skipped during test
                # discovery; real failures in parsing/assertions must surface.
                continue
            if not isinstance(data, dict):
                continue
            if data.get("metadata", {}).get("k_pivot_Mpc") is not None:
                data["_path"] = str(path)
                records.append(data)
        return records

    def run_cli(self, *argv, timeout: float = 900.0) -> subprocess.CompletedProcess[str]:
        """Run the real pipeline CLI through the harness with cwd in the sandbox."""
        env = os.environ.copy()
        env["PBH_SANDBOX_ROOT"] = str(self.root)
        return subprocess.run(
            [sys.executable, str(self.harness), *[str(a) for a in argv]],
            cwd=str(self.root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


@pytest.fixture
def pbh_pipeline_sandbox(tmp_path, monkeypatch):
    """Redirect PBH pipeline I/O into a throwaway sandbox (opt-in, not autouse).

    Patches the constants root, every plotting output directory, and the cwd
    for in-process calls, and provides a named CLI harness that applies the
    same redirection before executing the real script with ``runpy``. Module
    globals are restored by pytest's monkeypatch at teardown.
    """
    import scripts.constants as constants
    import scripts.plotting as plotting

    repo_root = Path(constants.ROOT_DIR).resolve()
    pipeline_script = repo_root / "scripts" / "full_pbh_pipeline.py"
    root = (tmp_path / "sandbox").resolve()
    root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(constants, "ROOT_DIR", str(root), raising=False)
    sandboxed_dirs = {
        key: str(root / Path(value).resolve().relative_to(repo_root))
        for key, value in plotting.OUTPUT_DIRS.items()
    }
    monkeypatch.setattr(plotting, "OUTPUT_DIRS", sandboxed_dirs, raising=False)
    monkeypatch.chdir(root)

    # Modules that bound ROOT_DIR with ``from scripts.constants import ROOT_DIR``
    # at import time keep the repo value unless their global is patched too.
    for mod_name in (
        "scripts.full_pbh_pipeline",
        "pspectrum_pipeline",
        "inf_dyn_MS_full",
    ):
        module = sys.modules.get(mod_name)
        if module is not None and hasattr(module, "ROOT_DIR"):
            monkeypatch.setattr(module, "ROOT_DIR", str(root), raising=False)

    harness = root / "pbh_cli_harness.py"
    harness.write_text(_HARNESS_SOURCE, encoding="utf-8")

    return PBHPipelineSandbox(
        root=root,
        harness=harness,
        repo_root=repo_root,
        pipeline_script=pipeline_script,
    )

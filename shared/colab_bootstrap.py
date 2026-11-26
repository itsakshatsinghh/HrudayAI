"""Reusable Google Colab and local repository bootstrap helper for HrudayAI.

Handles automatic cloning of the public GitHub repository into /content/HrudayAI
when notebooks are opened directly in Colab, verifies existing clones, optionally
pulls the latest branch, and configures sys.path and working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PUBLIC_REPO_URL = "https://github.com/itsakshatsinghh/HrudayAI.git"
DEFAULT_COLAB_ROOT = Path("/content/HrudayAI")
DEFAULT_BRANCH = "main"


class ColabBootstrapError(RuntimeError):
    """Raised when cloning, validating, or updating the HrudayAI repository fails."""


def _run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = ["git", *args]
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=True,
    )


def _is_valid_git_repo(repo_path: Path) -> bool:
    if not repo_path.is_dir() or not (repo_path / ".git").exists():
        return False
    try:
        res = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=repo_path)
        return res.stdout.strip().lower() == "true"
    except (subprocess.CalledProcessError, OSError):
        return False


def get_git_branch(repo_path: Path) -> str:
    """Return the active Git branch name for repo_path."""
    try:
        res = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
        branch = res.stdout.strip()
        return branch or "unknown"
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def get_git_status_summary(repo_path: Path) -> str:
    """Return a concise Git working-tree status string."""
    try:
        res = _run_git(["status", "--short"], cwd=repo_path)
        return res.stdout.strip() or "clean"
    except (subprocess.CalledProcessError, OSError) as exc:
        return f"unavailable ({exc})"


def find_local_repo_root(start_path: Path | None = None) -> Path:
    """Walk upward from start_path (or cwd) to locate the HrudayAI repository root."""
    current = (start_path or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if _is_valid_git_repo(candidate):
            return candidate
        if (candidate / "shared").is_dir() and (candidate / "phase1_voice").is_dir():
            return candidate
    return current


def ensure_sys_path(repo_root: Path) -> None:
    """Insert repo_root at the front of sys.path if not already present."""
    root_str = str(repo_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def bootstrap_repo(
    repo_url: str = PUBLIC_REPO_URL,
    colab_root: str | Path = DEFAULT_COLAB_ROOT,
    branch: str = DEFAULT_BRANCH,
    pull_latest: bool = True,
    in_colab: bool | None = None,
    change_dir: bool = True,
    start_path: Path | None = None,
) -> dict[str, Any]:
    """Ensure HrudayAI is available, configured on sys.path, and ready for notebook execution.

    In Google Colab:
      - If `colab_root` does not exist, clones the public GitHub repository (no token required).
      - If `colab_root` exists, validates it as a Git repository and optionally pulls `branch`.
    Locally:
      - Locates the repository root dynamically regardless of notebook working directory.
    """
    is_colab = ("google.colab" in sys.modules) if in_colab is None else bool(in_colab)
    action = "local"

    if is_colab:
        repo_path = Path(colab_root)
        if not repo_path.exists():
            try:
                _run_git(["clone", "--branch", branch, repo_url, str(repo_path)])
                action = "cloned"
            except (subprocess.CalledProcessError, OSError) as exc:
                stderr = getattr(exc, "stderr", "") or str(exc)
                raise ColabBootstrapError(
                    f"Failed to clone public repository '{repo_url}' into '{repo_path}'. "
                    f"Verify internet access in Colab and repository availability.\nDetails: {stderr.strip()}"
                ) from exc
        else:
            if not _is_valid_git_repo(repo_path):
                raise ColabBootstrapError(
                    f"Directory '{repo_path}' already exists but is not a valid Git repository. "
                    f"Remove or rename it (e.g. `!rm -rf {repo_path}`) and re-run this cell."
                )
            action = "reused"
            if pull_latest:
                try:
                    _run_git(["pull", "--ff-only", "origin", branch], cwd=repo_path)
                    action = "pulled"
                except (subprocess.CalledProcessError, OSError) as exc:
                    stderr = getattr(exc, "stderr", "") or str(exc)
                    raise ColabBootstrapError(
                        f"Repository exists at '{repo_path}', but pulling latest '{branch}' failed. "
                        f"Check `!git -C {repo_path} status` or network access.\nDetails: {stderr.strip()}"
                    ) from exc
    else:
        repo_path = find_local_repo_root(start_path=start_path)

    repo_path = repo_path.resolve()
    ensure_sys_path(repo_path)
    if change_dir and repo_path.is_dir():
        os.chdir(repo_path)

    active_branch = get_git_branch(repo_path)
    git_status = get_git_status_summary(repo_path)

    print(
        f"[HrudayAI Bootstrap] Ready at {repo_path} "
        f"(branch={active_branch}, mode={'colab' if is_colab else 'local'}, action={action})"
    )
    return {
        "repo_root": repo_path,
        "branch": active_branch,
        "in_colab": is_colab,
        "action": action,
        "git_status": git_status,
    }

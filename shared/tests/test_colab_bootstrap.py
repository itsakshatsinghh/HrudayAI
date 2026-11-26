"""Unit tests for shared.colab_bootstrap."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from shared.colab_bootstrap import (
    PUBLIC_REPO_URL,
    ColabBootstrapError,
    bootstrap_repo,
    find_local_repo_root,
)


class ColabBootstrapTests(unittest.TestCase):
    def test_missing_repo_in_colab_clones_public_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "HrudayAI"
            calls: list[list[str]] = []

            def fake_run(cmd, cwd=None, capture_output=True, text=True, check=True):
                calls.append(cmd)
                if cmd[:2] == ["git", "clone"]:
                    target.mkdir(parents=True, exist_ok=True)
                    (target / ".git").mkdir()
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                if "rev-parse" in cmd and "--abbrev-ref" in cmd:
                    return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
                if "status" in cmd:
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                return subprocess.CompletedProcess(cmd, 0, stdout="true\n", stderr="")

            with patch("shared.colab_bootstrap.subprocess.run", side_effect=fake_run):
                info = bootstrap_repo(
                    colab_root=target,
                    in_colab=True,
                    change_dir=False,
                    pull_latest=True,
                )

            self.assertEqual(info["action"], "cloned")
            self.assertEqual(info["branch"], "main")
            self.assertEqual(info["repo_root"], target.resolve())
            self.assertIn(str(target.resolve()), sys.path)
            self.assertEqual(
                calls[0],
                ["git", "clone", "--branch", "main", PUBLIC_REPO_URL, str(target)],
            )

    def test_existing_valid_repo_pulls_without_recloning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "HrudayAI"
            target.mkdir()
            (target / ".git").mkdir()
            calls: list[list[str]] = []

            def fake_run(cmd, cwd=None, capture_output=True, text=True, check=True):
                calls.append(cmd)
                if "--is-inside-work-tree" in cmd:
                    return subprocess.CompletedProcess(cmd, 0, stdout="true\n", stderr="")
                if "pull" in cmd:
                    return subprocess.CompletedProcess(cmd, 0, stdout="Already up to date.\n", stderr="")
                if "--abbrev-ref" in cmd:
                    return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("shared.colab_bootstrap.subprocess.run", side_effect=fake_run):
                info = bootstrap_repo(
                    colab_root=target,
                    in_colab=True,
                    change_dir=False,
                    pull_latest=True,
                )

            self.assertEqual(info["action"], "pulled")
            self.assertFalse(any(c[:2] == ["git", "clone"] for c in calls))
            self.assertTrue(any("pull" in c for c in calls))

    def test_existing_non_git_directory_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "HrudayAI"
            target.mkdir()
            with self.assertRaises(ColabBootstrapError) as ctx:
                bootstrap_repo(colab_root=target, in_colab=True, change_dir=False)
            self.assertIn("not a valid Git repository", str(ctx.exception))

    def test_clone_failure_raises_bootstrap_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "MissingRepo"
            err = subprocess.CalledProcessError(128, ["git", "clone"], stderr="fatal: network error")
            with patch("shared.colab_bootstrap.subprocess.run", side_effect=err):
                with self.assertRaises(ColabBootstrapError) as ctx:
                    bootstrap_repo(colab_root=target, in_colab=True, change_dir=False)
            self.assertIn("Failed to clone public repository", str(ctx.exception))

    def test_find_local_repo_root_from_nested_notebook_dir(self) -> None:
        nested = Path(__file__).resolve().parents[2] / "phase2_symptom_extraction" / "notebooks"
        root = find_local_repo_root(start_path=nested)
        self.assertTrue((root / "shared" / "colab_bootstrap.py").exists())


if __name__ == "__main__":
    unittest.main()

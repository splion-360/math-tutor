"""Test evidence integrity, download boundaries, and missing-artifact handling.
The cases ensure plotting cannot silently use absent or altered numeric results."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest

from dynamic_lora import evidence


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Response(io.BytesIO):
    def geturl(self) -> str:
        return "https://example.com/artifact.json"


def test_verified_download_is_reused_without_network(tmp_path: Path, monkeypatch: Any) -> None:
    """Matching existing bytes require no network access."""
    destination = tmp_path / "data.json"
    destination.write_bytes(b"observed")

    def unexpected_request(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a matching artifact should not be downloaded again")

    monkeypatch.setattr(evidence, "urlopen", unexpected_request)
    evidence.download_verified("https://example.com/data", destination, _sha(b"observed"))


def test_download_publishes_only_matching_bytes(tmp_path: Path, monkeypatch: Any) -> None:
    """A successful download preserves the original bytes exactly."""
    monkeypatch.setattr(evidence, "urlopen", lambda *a, **k: _Response(b"observed"))
    destination = tmp_path / "data.json"
    evidence.download_verified("https://example.com/data", destination, _sha(b"observed"))
    assert destination.read_bytes() == b"observed"


def test_altered_download_leaves_no_artifact(tmp_path: Path, monkeypatch: Any) -> None:
    """Hash failures leave no file that could be mistaken for verified evidence."""
    monkeypatch.setattr(evidence, "urlopen", lambda *a, **k: _Response(b"altered"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evidence.download_verified("https://example.com/data", tmp_path / "data", _sha(b"original"))
    assert list(tmp_path.iterdir()) == []


def test_existing_corruption_is_reported(tmp_path: Path) -> None:
    """Local corruption is surfaced instead of silently replacing its bytes."""
    destination = tmp_path / "data"
    destination.write_bytes(b"altered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evidence.download_verified("https://example.com/data", destination, _sha(b"original"))
    assert destination.read_bytes() == b"altered"


@pytest.mark.parametrize("path", ["../outside", "/outside", "", "."])
def test_manifest_paths_cannot_escape(tmp_path: Path, path: str) -> None:
    """Unsafe paths cannot overwrite files outside the evidence directory."""
    with pytest.raises(ValueError):
        evidence.artifact_path(tmp_path, path)


def test_manifest_symlinks_cannot_escape(tmp_path: Path) -> None:
    """A symlink is subject to the same destination boundary as a relative path."""
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        evidence.artifact_path(tmp_path, "escape/file")


@pytest.mark.parametrize("url", ["http://example.com/data", "file:///tmp/data", "https://u:p@example.com/data"])
def test_private_or_non_https_urls_are_rejected(tmp_path: Path, url: str) -> None:
    """Evidence retrieval accepts public HTTPS URLs without embedded credentials."""
    with pytest.raises(ValueError, match="public HTTPS"):
        evidence.download_verified(url, tmp_path / "data", _sha(b"original"))


def test_download_size_is_bounded(tmp_path: Path, monkeypatch: Any) -> None:
    """Oversized responses stop before an artifact is published."""
    monkeypatch.setattr(evidence, "_MAX_DOWNLOAD_BYTES", 3)
    monkeypatch.setattr(evidence, "urlopen", lambda *a, **k: _Response(b"large"))
    with pytest.raises(ValueError, match="size limit"):
        evidence.download_verified("https://example.com/data", tmp_path / "data", _sha(b"large"))
    assert list(tmp_path.iterdir()) == []


def test_verify_reports_missing_original_files(tmp_path: Path) -> None:
    """Unavailable originals cannot pass validation or be inferred from a figure."""
    manifest = {"artifacts": [{"path": "summary.json", "sha256": _sha(b"original")}]}
    with pytest.raises(ValueError, match="missing original evidence: summary.json"):
        evidence.verify_artifacts(manifest, tmp_path)


def test_verify_rejects_duplicate_evidence(tmp_path: Path) -> None:
    """A duplicated manifest entry cannot stand in for a second observed artifact."""
    entry = {"path": "summary.json", "sha256": _sha(b"original")}
    with pytest.raises(ValueError, match="duplicate"):
        evidence.verify_artifacts({"artifacts": [entry, entry]}, tmp_path)


def test_figures_require_verified_numeric_inputs(tmp_path: Path) -> None:
    """Figure generation checks integrity before importing plotting dependencies."""
    manifest = {"artifacts": [{"path": "summary.json", "sha256": _sha(b"original")}]}
    with pytest.raises(ValueError, match="missing original"):
        evidence.render_figures(manifest, tmp_path, tmp_path)
    assert not (tmp_path / "figures").exists()

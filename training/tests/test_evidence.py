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


def _dataset_fixture(tmp_path: Path, monkeypatch: Any) -> evidence.EvidenceManifest:
    """Provide a small corpus and CPU conversion seam for split-publication checks."""
    import json

    rows = [
        {"id": f"{difficulty}-{index}", "difficulty": difficulty}
        for difficulty in ("foundational", "intermediate", "advanced")
        for index in range(2)
    ]
    prepared = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "source.parquet").write_bytes(b"parquet")

    def convert(command: list[str], *, check: bool) -> None:
        assert check
        Path(command[command.index("--internal-output") + 1]).write_bytes(prepared)

    monkeypatch.setattr(evidence.subprocess, "run", convert)
    return {
        "schema_version": 1,
        "dataset": {
            "url": "https://example.com/source.parquet",
            "parquet_sha256": _sha(b"parquet"),
            "parquet_size_bytes": 7,
            "prepared_sha256": _sha(prepared),
            "prepared_records": 6,
            "seed": 42,
            "epochs": 3,
            "training_count": 3,
            "validation_count": 3,
        },
        "artifacts": [],
    }


def test_dataset_publishes_disjoint_reconstructed_ids(tmp_path: Path, monkeypatch: Any) -> None:
    """A matching corpus publishes all IDs once and labels their reconstructed status."""
    import json

    manifest = _dataset_fixture(tmp_path, monkeypatch)
    root = tmp_path / "evidence"
    evidence.prepare_dataset(manifest, root, tmp_path)
    split = json.loads((root / "split.json").read_text())
    assert len(split["training_ids"]) == len(split["validation_ids"]) == 3
    assert not set(split["training_ids"]) & set(split["validation_ids"])
    assert len(set(split["training_ids"] + split["validation_ids"])) == 6
    assert split["source_sha256"] == manifest["dataset"]["prepared_sha256"]
    assert split["status"] == "reconstructed_pending_saved_metadata_comparison"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parquet_size_bytes", 8, "Parquet size"),
        ("prepared_sha256", _sha(b"different"), "SHA-256 mismatch"),
        ("training_count", 4, "split counts"),
        ("prepared_records", 7, "split counts"),
    ],
)
def test_dataset_rejects_mismatches_before_split_publication(
    tmp_path: Path, monkeypatch: Any, field: str, value: Any, message: str
) -> None:
    """Corrupt source identity or changed coverage cannot publish a valid-looking split."""
    manifest = _dataset_fixture(tmp_path, monkeypatch)
    manifest["dataset"][field] = value
    root = tmp_path / "evidence"
    with pytest.raises(ValueError, match=message):
        evidence.prepare_dataset(manifest, root, tmp_path)
    assert not (root / "split.json").exists()


def test_missing_manifest_is_a_cli_error(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    """A missing input manifest produces a concise expected error rather than a traceback."""
    monkeypatch.setattr(
        evidence.sys, "argv", ["evidence", "verify", "--manifest", str(tmp_path / "missing.json")]
    )
    with pytest.raises(SystemExit) as error:
        evidence.main()
    assert error.value.code == 1
    assert "Evidence check failed:" in capsys.readouterr().err


def test_dataset_invalidates_stale_split_on_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """A failed rebuild cannot leave a previous split looking like the new result."""
    manifest = _dataset_fixture(tmp_path, monkeypatch)
    manifest["dataset"]["training_count"] = 4
    root = tmp_path / "evidence"
    (root / "split.json").write_text('{"status":"old"}')
    with pytest.raises(ValueError, match="split counts"):
        evidence.prepare_dataset(manifest, root, tmp_path)
    assert not (root / "split.json").exists()


def test_manifest_location_does_not_change_repository(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A copied manifest still uses the explicitly selected checkout for conversion."""
    import json

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "dataset": {}, "artifacts": []}))
    repository = tmp_path / "checkout"
    selected: list[Path] = []
    monkeypatch.setattr(evidence, "prepare_dataset", lambda m, r, repo: selected.append(repo))
    monkeypatch.setattr(evidence.sys, "argv", [
        "evidence", "dataset", "--manifest", str(manifest_path), "--repository", str(repository)
    ])
    evidence.main()
    assert selected == [repository.resolve()]

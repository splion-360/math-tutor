"""Download and verify the numeric evidence used by Manim experiment figures.
The module preserves recorded hashes and rejects incomplete or altered evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from dynamic_lora.full_corpus_plan import build_full_corpus_plan

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024


def file_sha256(path: Path) -> str:
    """Hash an artifact without loading it into memory.

    Args:
        path: Artifact file to read.

    Returns:
        Lowercase hexadecimal SHA-256 of the original bytes.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> None:
    """Reject absent files or bytes that differ from their recorded hash.

    Args:
        path: Local artifact to verify.
        expected_sha256: Recorded lowercase hexadecimal SHA-256.

    Raises:
        ValueError: If the recorded hash is invalid or the content differs.
        OSError: If the file cannot be read.
    """
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError(f"invalid SHA-256 for {path.name}")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path.name}")


def download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    """Download a public HTTPS artifact and publish it locally after verification.

    Args:
        url: Public HTTPS file URL without embedded credentials.
        destination: Local file path; existing matching files are reused.
        expected_sha256: Hash of the original artifact bytes.

    Raises:
        ValueError: If the URL, hash, download size, or content is invalid.
        OSError: If a network or filesystem operation fails.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("artifact URL must be public HTTPS without embedded credentials")
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError("artifact needs a recorded SHA-256 before downloading")
    if destination.exists():
        verify_file(destination, expected_sha256)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary_dir:
        temporary = Path(temporary_dir) / "download"
        with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            if urlparse(response.geturl()).scheme != "https":
                raise ValueError("artifact redirect must remain HTTPS")
            total = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    raise ValueError("numeric evidence exceeds the download size limit")
                stream.write(chunk)
        verify_file(temporary, expected_sha256)
        temporary.replace(destination)


def artifact_path(root: Path, relative_path: str) -> Path:
    """Resolve a manifest path while rejecting escape through paths or symlinks.

    Args:
        root: Evidence directory.
        relative_path: File path recorded in the manifest.

    Returns:
        Resolved path within the evidence directory.

    Raises:
        ValueError: If the path is empty, absolute, or escapes the directory.
    """
    selected = Path(relative_path)
    if not relative_path or selected.is_absolute() or ".." in selected.parts:
        raise ValueError("artifact path must be a relative file path without traversal")
    path = (root / selected).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("artifact path escapes the evidence directory")
    return path


def verify_artifacts(manifest: dict[str, Any], root: Path) -> None:
    """Require every numeric artifact to match its recorded hash.

    Args:
        manifest: Evidence manifest with an artifacts list.
        root: Directory containing the downloaded artifacts.

    Raises:
        ValueError: If artifacts are absent, invalid, or duplicated.
        OSError: If an artifact cannot be read.
    """
    artifacts = manifest["artifacts"]
    if not artifacts:
        raise ValueError("manifest has no numeric evidence")
    paths = [artifact_path(root, entry["path"]) for entry in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate artifact paths in manifest")
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError("missing original evidence: " + ", ".join(missing))
    for entry, path in zip(artifacts, paths, strict=True):
        verify_file(path, entry["sha256"])


def prepare_dataset(manifest: dict[str, Any], root: Path, repository: Path) -> None:
    """Rebuild the original corpus and write a deterministic split for comparison.

    Args:
        manifest: Frozen dataset identity and split settings.
        root: Destination for source data and reconstructed split IDs.
        repository: Checkout containing the existing preparation script.

    Raises:
        ValueError: If corpus bytes or split counts differ from recorded expectations.
        OSError: If an input, download, or output cannot be accessed.
        subprocess.CalledProcessError: If CPU dataset conversion fails.
    """
    dataset = manifest["dataset"]
    parquet = root / "source.parquet"
    download_verified(dataset["url"], parquet, dataset["parquet_sha256"])
    if parquet.stat().st_size != dataset["parquet_size_bytes"]:
        raise ValueError("dataset Parquet size differs from the pinned source")
    prepared = root / "bespoke_manim_train.jsonl"
    subprocess.run(
        [
            sys.executable, str(repository / "training/scripts/prepare_bespoke_manim.py"),
            str(parquet), "--internal-output", str(prepared),
            "--nebius-output", str(root / "messages.jsonl"),
        ],
        check=True,
    )
    verify_file(prepared, dataset["prepared_sha256"])
    records = [json.loads(line) for line in prepared.read_text().splitlines() if line.strip()]
    plan = build_full_corpus_plan(records, seed=dataset["seed"], epochs=dataset["epochs"])
    if (len(records), len(plan.training_ids), len(plan.validation_ids)) != (
        dataset["prepared_records"], dataset["training_count"], dataset["validation_count"]
    ):
        raise ValueError("reconstructed corpus or split counts differ from the recorded run")
    split = {
        "status": "reconstructed_pending_saved_metadata_comparison",
        "source_sha256": dataset["prepared_sha256"],
        "seed": dataset["seed"],
        "training_ids": list(plan.training_ids),
        "validation_ids": list(plan.validation_ids),
    }
    (root / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")


def render_figures(manifest: dict[str, Any], root: Path, repository: Path) -> None:
    """Regenerate the reported plots after checking the original numeric artifacts.

    Args:
        manifest: Frozen numeric evidence manifest.
        root: Directory containing summaries and receiving the figures directory.
        repository: Checkout containing the original plotting scripts.

    Raises:
        ValueError: If evidence is missing, altered, or incompatible with the plots.
        OSError: If files cannot be read or figures cannot be written.
    """
    verify_artifacts(manifest, root)
    import matplotlib

    matplotlib.use("Agg")
    scripts = repository / "training/scripts"
    heatmap = runpy.run_path(str(scripts / "plot_validation_gradient_heatmap.py"))["plot_heatmap"]
    cosines = runpy.run_path(str(scripts / "plot_validation_gradient_cosines.py"))["plot_cosines"]
    heatmap(root / "base_norms.json", root / "figures/base_norms.png")
    heatmap(root / "lora_norms.json", root / "figures/lora_norms.png")
    cosines(root / "lora_cosines.json", root / "figures/cosines")


def main() -> None:
    """Run CPU dataset reconstruction, evidence retrieval, verification, or plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dataset", "fetch", "verify", "figures"))
    parser.add_argument("--manifest", type=Path, default=Path("training/evidence/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("training/artifacts/evidence"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    repository = args.manifest.resolve().parents[2]
    try:
        if args.command == "dataset":
            prepare_dataset(manifest, args.output, repository)
        elif args.command == "fetch":
            unpublished = [entry["path"] for entry in manifest["artifacts"] if not entry["url"]]
            if unpublished:
                raise ValueError("public artifact URLs unavailable: " + ", ".join(unpublished))
            for entry in manifest["artifacts"]:
                download_verified(
                    entry["url"], artifact_path(args.output, entry["path"]), entry["sha256"]
                )
            verify_artifacts(manifest, args.output)
        elif args.command == "verify":
            verify_artifacts(manifest, args.output)
        else:
            render_figures(manifest, args.output, repository)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Evidence check failed: {error}\n")
    print(f"Evidence {args.command} completed: {args.output}")


if __name__ == "__main__":
    main()

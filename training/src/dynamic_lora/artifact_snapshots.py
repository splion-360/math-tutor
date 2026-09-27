"""Persist diagnostic snapshots and manifests produced during training.
The store owns deterministic JSON serialization and artifact paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SnapshotStore:
    """Write one JSONL snapshot stream and its JSON manifest."""

    def __init__(self, artifact_dir: Path, *, name: str) -> None:
        self.snapshot_path = artifact_dir / f"{name}.jsonl"
        self.manifest_path = artifact_dir / f"{name}_manifest.json"
        self._initialized = False

    def append(self, records: list[dict[str, Any]]) -> None:
        """Append records, replacing an older stream on the first write."""
        if not records:
            return
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._initialized else "w"
        with self.snapshot_path.open(mode, encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._initialized = True

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """Write the latest summary for the snapshot stream."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

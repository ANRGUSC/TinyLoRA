from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointRecord:
    run_id: str
    step: int
    checkpoint_dir: Path
    state_path: Path
    manifest_path: Path
    completed_marker: Path
    created_at: str


class CheckpointManager:
    """
    Atomic, resumable checkpoint manager.

    Safety design:
    - Never deletes checkpoints; archives old ones under `archive/`.
    - Writes temporary content first, then promotes to final directory.
    - Marks completion only after state + manifest are valid.
    """

    def __init__(self, root_dir: str | Path, keep_last_n: int = 8) -> None:
        self.root_dir = Path(root_dir)
        self.keep_last_n = max(1, int(keep_last_n))

    def _run_root(self, run_id: str) -> Path:
        return self.root_dir / "runs" / run_id

    def _checkpoints_root(self, run_id: str) -> Path:
        return self._run_root(run_id) / "checkpoints"

    def _archive_root(self, run_id: str) -> Path:
        return self._run_root(run_id) / "archive"

    def _manifests_root(self, run_id: str) -> Path:
        return self._run_root(run_id) / "manifests"

    def _checkpoint_name(self, step: int) -> str:
        return f"step_{step:09d}"

    def _ensure_roots(self, run_id: str) -> tuple[Path, Path, Path, Path]:
        run_root = self._run_root(run_id)
        ckpt_root = self._checkpoints_root(run_id)
        manifests_root = self._manifests_root(run_id)
        archive_root = self._archive_root(run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        ckpt_root.mkdir(parents=True, exist_ok=True)
        manifests_root.mkdir(parents=True, exist_ok=True)
        archive_root.mkdir(parents=True, exist_ok=True)
        return run_root, ckpt_root, manifests_root, archive_root

    def _write_latest_manifest(self, run_id: str, final_dir: Path, created_at: str, step: int) -> None:
        manifests_root = self._manifests_root(run_id)
        latest_manifest = {
            "run_id": run_id,
            "step": step,
            "checkpoint_dir": str(final_dir),
            "created_at": created_at,
        }
        latest_path = manifests_root / "latest_checkpoint.json"
        tmp_latest = manifests_root / f".tmp_latest_{uuid.uuid4().hex[:8]}.json"
        with tmp_latest.open("w", encoding="utf-8") as handle:
            json.dump(latest_manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
        tmp_latest.replace(latest_path)

    def _promote_temp_dir(self, run_id: str, final_dir: Path, temp_dir: Path) -> None:
        archive_root = self._archive_root(run_id)
        if final_dir.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archived_name = f"{final_dir.name}_replaced_{timestamp}"
            shutil.move(str(final_dir), str(archive_root / archived_name))
        temp_dir.rename(final_dir)

    def save_files_checkpoint(
        self,
        run_id: str,
        step: int,
        files: dict[str, bytes],
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        _, ckpt_root, _, _ = self._ensure_roots(run_id)
        created_at = utc_now_iso()
        final_dir = ckpt_root / self._checkpoint_name(step)
        temp_dir = ckpt_root / f".tmp_{self._checkpoint_name(step)}_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=False)

        manifest_files: list[dict[str, str]] = []
        for rel_path, blob in files.items():
            clean_rel = rel_path.replace("\\", "/")
            if clean_rel.startswith("../") or clean_rel.startswith("/"):
                raise ValueError(f"Unsafe checkpoint file path: {rel_path}")
            out_path = temp_dir / clean_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob)
            manifest_files.append({"path": clean_rel, "sha256": sha256_file(out_path)})

        manifest = {
            "run_id": run_id,
            "step": step,
            "created_at": created_at,
            "files": manifest_files,
            "metadata": metadata or {},
        }
        manifest_path = temp_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)

        completed_marker = temp_dir / "COMPLETED"
        completed_marker.write_text(created_at, encoding="utf-8")

        self._promote_temp_dir(run_id, final_dir=final_dir, temp_dir=temp_dir)
        self._write_latest_manifest(run_id=run_id, final_dir=final_dir, created_at=created_at, step=step)
        self._archive_older_checkpoints(run_id)

        return CheckpointRecord(
            run_id=run_id,
            step=step,
            checkpoint_dir=final_dir,
            state_path=final_dir / "state.json",
            manifest_path=final_dir / "manifest.json",
            completed_marker=final_dir / "COMPLETED",
            created_at=created_at,
        )

    def save_json_checkpoint(
        self, run_id: str, step: int, state: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> CheckpointRecord:
        payload = json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        return self.save_files_checkpoint(
            run_id=run_id,
            step=step,
            files={"state.json": payload},
            metadata=metadata,
        )

    def _archive_older_checkpoints(self, run_id: str) -> None:
        ckpt_root = self._checkpoints_root(run_id)
        archive_root = self._archive_root(run_id)
        checkpoints = sorted(
            [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")],
            key=lambda p: p.name,
        )
        if len(checkpoints) <= self.keep_last_n:
            return
        to_archive = checkpoints[: len(checkpoints) - self.keep_last_n]
        for path in to_archive:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            target = archive_root / f"{path.name}_archived_{timestamp}"
            if target.exists():
                target = archive_root / f"{path.name}_archived_{timestamp}_{uuid.uuid4().hex[:6]}"
            shutil.move(str(path), str(target))

    def validate_checkpoint_dir(self, checkpoint_dir: str | Path) -> bool:
        checkpoint_path = Path(checkpoint_dir)
        manifest_path = checkpoint_path / "manifest.json"
        completed_marker = checkpoint_path / "COMPLETED"
        if not manifest_path.exists() or not completed_marker.exists():
            return False

        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            files = manifest["files"]
        except Exception:
            return False

        for file_entry in files:
            rel = file_entry.get("path")
            expected = file_entry.get("sha256")
            if not rel or not expected:
                return False
            file_path = checkpoint_path / rel
            if not file_path.exists():
                return False
            if sha256_file(file_path) != expected:
                return False
        return True

    def find_latest_valid_checkpoint(self, run_id: str) -> CheckpointRecord | None:
        manifests_root = self._manifests_root(run_id)
        latest_path = manifests_root / "latest_checkpoint.json"
        candidates: list[Path] = []

        if latest_path.exists():
            try:
                with latest_path.open("r", encoding="utf-8") as handle:
                    latest = json.load(handle)
                candidates.append(Path(latest["checkpoint_dir"]))
            except Exception:
                pass

        ckpt_root = self._checkpoints_root(run_id)
        if ckpt_root.exists():
            all_dirs = sorted(
                [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")],
                key=lambda p: p.name,
                reverse=True,
            )
            candidates.extend([p for p in all_dirs if p not in candidates])

        for candidate in candidates:
            if self.validate_checkpoint_dir(candidate):
                step = int(candidate.name.split("_")[1])
                with (candidate / "manifest.json").open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                return CheckpointRecord(
                    run_id=run_id,
                    step=step,
                    checkpoint_dir=candidate,
                    state_path=candidate / "state.json",
                    manifest_path=candidate / "manifest.json",
                    completed_marker=candidate / "COMPLETED",
                    created_at=manifest["created_at"],
                )
        return None

    def load_latest_state(self, run_id: str) -> dict[str, Any] | None:
        latest = self.find_latest_valid_checkpoint(run_id)
        if latest is None:
            return None
        with latest.state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

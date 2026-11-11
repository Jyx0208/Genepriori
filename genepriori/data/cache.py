"""Local cache for downloaded data and fetched evidence.

Everything lives under ``$GENEPRIORI_HOME`` (default ``~/.genepriori``) so
that repeated runs do not re-download Open Targets pages or the STRING
network. Every cached artifact is registered in ``manifest.json`` together
with its source URL, fetch time, size and checksum.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

HOME_ENV = "GENEPRIORI_HOME"
MANIFEST_NAME = "manifest.json"

PathLike = Union[str, "os.PathLike[str]"]


def default_home() -> Path:
    """Root data directory, overridable with ``GENEPRIORI_HOME``."""
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".genepriori"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: PathLike, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class Cache:
    """Filesystem cache rooted at ``<home>/cache`` by default."""

    def __init__(self, root: Optional[PathLike] = None) -> None:
        if root is None:
            root = default_home() / "cache"
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def exists(self, *parts: str) -> bool:
        return self.path(*parts).exists()

    def ensure_parent(self, *parts: str) -> Path:
        p = self.path(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------- json
    def read_json(self, *parts: str) -> Any:
        with open(self.path(*parts), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_json(self, obj: Any, *parts: str) -> Path:
        path = self.ensure_parent(*parts)
        payload = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
        _atomic_write(path, payload.encode("utf-8"))
        return path

    def write_bytes(self, data: bytes, *parts: str) -> Path:
        path = self.ensure_parent(*parts)
        _atomic_write(path, data)
        return path

    # --------------------------------------------------------------- manifest
    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def manifest(self) -> Dict[str, Dict[str, Any]]:
        if not self.manifest_path.exists():
            return {}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def record(
        self,
        key: str,
        *,
        url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        compute_sha256: bool = True,
    ) -> Dict[str, Any]:
        """Register a cached artifact; returns the stored entry."""
        path = self.path(key)
        entry: Dict[str, Any] = {
            "path": key,
            "fetched_at": utc_now(),
            "size": path.stat().st_size if path.exists() else None,
            "url": url,
        }
        if compute_sha256 and path.exists() and path.is_file():
            entry["sha256"] = sha256_file(path)
        if extra:
            entry.update(extra)
        manifest = self.manifest()
        manifest[key] = entry
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        _atomic_write(self.manifest_path, payload.encode("utf-8"))
        return entry

    def info(self, key: str) -> Optional[Dict[str, Any]]:
        return self.manifest().get(key)

    def is_fresh(self, key: str, *, max_age_days: Optional[float] = None) -> bool:
        """Whether a cached file exists (and is younger than ``max_age_days``)."""
        if not self.path(key).exists():
            return False
        if max_age_days is None:
            return True
        entry = self.info(key) or {}
        stamp = entry.get("fetched_at")
        if not stamp:
            return False
        try:
            fetched = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            return False
        age = (datetime.now(timezone.utc) - fetched).total_seconds() / 86400.0
        return age <= max_age_days

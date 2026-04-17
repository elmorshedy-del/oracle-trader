from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


UTC = timezone.utc


def register_log_namespace(
    *,
    root: Path,
    lane_key: str,
    label: str,
    category: str,
    source: str,
    description: str,
    paths: dict[str, Path],
    extra: dict[str, object] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    now = datetime.now(UTC).isoformat()
    existing = _load_json(manifest_path)
    manifest = {
        "lane_key": lane_key,
        "label": label,
        "category": category,
        "source": source,
        "description": description,
        "root": str(root),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "important_files": {key: str(value) for key, value in sorted(paths.items())},
        "extra": extra or {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _update_catalog(root.parent / "_catalog.json", manifest)


def _update_catalog(catalog_path: Path, manifest: dict[str, object]) -> None:
    catalog = _load_json(catalog_path)
    entries = catalog if isinstance(catalog, list) else []
    kept: list[dict[str, object]] = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("lane_key") != manifest["lane_key"]:
            kept.append(entry)
    kept.append(
        {
            "lane_key": manifest["lane_key"],
            "label": manifest["label"],
            "category": manifest["category"],
            "source": manifest["source"],
            "root": manifest["root"],
            "manifest": str(Path(manifest["root"]) / "manifest.json"),
            "updated_at": manifest["updated_at"],
        }
    )
    kept.sort(key=lambda entry: str(entry.get("lane_key") or ""))
    catalog_path.write_text(json.dumps(kept, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, object] | list[object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

"""A-00 read-only local data/rule inventory; it never claims economic eligibility."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any
from urllib.parse import quote
import sqlite3


def build_inventory(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1"],
                            check=True, capture_output=True, text=True).stdout.splitlines()
    data_root = root / "data"
    data_sets = []
    if data_root.is_dir():
        for path in sorted(p for p in data_root.iterdir() if p.is_dir()):
            files = [p for p in path.rglob("*") if p.is_file()]
            data_sets.append({"name": path.name, "file_count": len(files),
                              "bytes": sum(p.stat().st_size for p in files)})
    rules_candidates = sorted(
        str(path.relative_to(root))
        for base in (root / "configs", root / "data", root / "tests" / "fixtures")
        if base.is_dir()
        for path in base.rglob("*.json")
        if any(term in path.name.lower() for term in ("rule", "instrument"))
    )
    evidence_paths = [
        "docs/HISTORICAL_RULES_BLOCKER_20261016.md",
        "docs/AUDIT_REPORT_20261016.md",
        "docs/audit_evidence/20261016/persistent_artifacts_v2.json",
    ]
    registry_path = root / ".artifacts" / "research-access.sqlite"
    registry: dict[str, Any] = {"available": registry_path.is_file(), "access_count": None,
                                "intervals": [], "purpose_state_counts": {},
                                "query_intent_table_available": False}
    if registry_path.is_file():
        uri = "file:" + quote(registry_path.as_posix(), safe="/:\\") + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            )}
            registry["query_intent_table_available"] = "query_intent" in tables
            if "access" in tables:
                rows = connection.execute(
                    "SELECT access_key,market,family,start_ns,end_ns,purpose,state "
                    "FROM access ORDER BY market,family,start_ns,end_ns,access_key",
                ).fetchall()
                registry["access_count"] = len(rows)
                registry["intervals"] = [
                    {"key": row[0], "market": row[1], "family": row[2],
                     "start_ns": row[3], "end_ns": row[4], "purpose": row[5], "state": row[6]}
                    for row in rows
                ]
                counts: dict[str, int] = {}
                for row in rows:
                    group = f"{row[5]}:{row[6]}"
                    counts[group] = counts.get(group, 0) + 1
                registry["purpose_state_counts"] = counts
        registry["sha256"] = sha256(registry_path.read_bytes()).hexdigest()
        registry["bytes"] = registry_path.stat().st_size

    archive_manifest_path = root / "docs" / "audit_evidence" / "20261016" / "persistent_artifacts_v2.json"
    archive_verification: dict[str, Any] = {
        "manifest_available": archive_manifest_path.is_file(),
        "verified_files": [], "missing_or_mismatched": [],
    }
    if archive_manifest_path.is_file():
        manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
        archive_root = root / ".artifacts" / "audit-development-20261016"
        for entry in manifest.get("files", []):
            file = archive_root / entry["path"]
            digest = sha256()
            if file.is_file():
                with file.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            if (file.is_file() and file.stat().st_size == entry["bytes"]
                    and digest.hexdigest() == entry["sha256"]):
                archive_verification["verified_files"].append(entry["path"])
            else:
                archive_verification["missing_or_mismatched"].append(entry["path"])
    return {
        "schema": "rles.il.a00_inventory.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "git_worktree_change_count": len(status),
        "working_tree_includes_uncommitted_changes": bool(status),
        "local_data_directories": data_sets,
        "historical_rule_file_candidates": rules_candidates,
        "economic_gate": {
            "eligible": False,
            "reason": "The existing audit marks historical ETHUSDT rules, account fees and margin history scenario-only or unverified.",
            "allowed": ["synthetic mechanism validation", "explicitly labeled Train-inner engineering diagnostics"],
            "blocked": ["formal historical net-return claims", "Test/Holdout strategy selection"],
        },
        "prior_evidence": {str(path): (root / path).is_file() for path in evidence_paths},
        "project_research_registry": registry,
        "persistent_archive_integrity": archive_verification,
        "research_registry_scope_note": (
            "The registry is append-only local governance, not a complete record of every historical data read. "
            "The 2026-10-16 inventory imported 67 known records from 10 WFO databases and 22 reports; "
            "older undocumented reads cannot be reconstructed from files alone."
        ),
        "rules_and_fee_scope": "No current exchange snapshot is substituted for historical effective rules.",
    }


def write_inventory(root: str | Path, output: str | Path | None = None) -> Path:
    root_path = Path(root).resolve()
    target = Path(output) if output else root_path / "docs" / "il" / "A00_inventory.json"
    if not target.is_absolute():
        target = root_path / target
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_inventory(root_path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = target.with_suffix(".md")
    data_lines = [
        f"- `{row['name']}`: {row['file_count']} files, {row['bytes']} bytes"
        for row in payload["local_data_directories"]
    ] or ["- No local data directories found."]
    status_lines = [f"- {key}: {'present' if value else 'missing'}"
                    for key, value in payload["prior_evidence"].items()]
    registry = payload["project_research_registry"]
    access_lines = [f"- `{key}`: {value}"
                    for key, value in sorted(registry["purpose_state_counts"].items())]
    archive = payload["persistent_archive_integrity"]
    markdown.write_text(
        "# A-00 data and rule inventory\n\n"
        f"Generated UTC: {payload['generated_at_utc']}\n\n"
        f"- Git HEAD: `{payload['git_head']}`\n"
        f"- Worktree changes: {payload['git_worktree_change_count']}\n"
        f"- Historical economics eligible: **{payload['economic_gate']['eligible']}**\n"
        f"- Gate reason: {payload['economic_gate']['reason']}\n\n"
        "## Local data directories\n\n" + "\n".join(data_lines) + "\n\n"
        "## Rule-file candidates\n\n"
        + ("\n".join(f"- `{name}`" for name in payload["historical_rule_file_candidates"])
           or "No matching rule JSON files found.") + "\n\n"
        "## Existing audit evidence\n\n" + "\n".join(status_lines) + "\n\n"
        "## Registered research intervals\n\n"
        f"- Registry available: {registry['available']}\n"
        f"- Access records: {registry['access_count']}\n"
        + ("\n".join(access_lines) or "No project registry records found.") + "\n"
        f"- Exact UTC-nanosecond intervals, purposes and states are in `{target.with_suffix('.json').name}`.\n"
        f"- Scope limit: {payload['research_registry_scope_note']}\n\n"
        "## Archived market/rule evidence\n\n"
        f"- Manifest available: {archive['manifest_available']}\n"
        f"- Hash verified files: {len(archive['verified_files'])}\n"
        f"- Missing or mismatched files: {len(archive['missing_or_mismatched'])}\n\n"
        "## Decision\n\n"
        "Proceed only with synthetic mechanism validation and clearly labeled Train-inner engineering diagnostics. "
        "Do not claim historical economic performance or select models on Test/Holdout. Current exchange rules "
        "are not substituted for historical effective rules.\n",
        encoding="utf-8",
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(write_inventory(args.root, args.output))


if __name__ == "__main__":
    main()

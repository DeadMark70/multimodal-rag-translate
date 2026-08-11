"""Normalize legacy document paths with an explicit dry-run/apply workflow."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.uploads import normalize_document_storage_path
from pdfserviceMD.repository import (
    list_document_path_rows,
    update_owned_document_paths,
)

_PATH_FIELDS = ("original_path", "translated_path")
_MIN_BATCH_SIZE = 1
_MAX_BATCH_SIZE = 1000


@dataclass(frozen=True)
class MigrationSummary:
    """Aggregate counts for one document-path migration run."""

    scanned_rows: int
    changed_fields: int
    applied_fields: int
    unchanged_fields: int
    rejected_fields: int


async def migrate_document_paths(
    *, apply: bool, batch_size: int = 100
) -> MigrationSummary:
    """Classify legacy paths and optionally update convertible fields."""
    if not _MIN_BATCH_SIZE <= batch_size <= _MAX_BATCH_SIZE:
        raise ValueError("batch_size must be between 1 and 1000")

    scanned_rows = 0
    changed_fields = 0
    applied_fields = 0
    unchanged_fields = 0
    rejected_fields = 0
    offset = 0

    while True:
        rows = await list_document_path_rows(offset=offset, limit=batch_size)
        if not rows:
            break

        for row in rows:
            scanned_rows += 1
            doc_id = row["id"]
            user_id = row["user_id"]
            pending: dict[str, str] = {}

            for field in _PATH_FIELDS:
                stored_path = row.get(field)
                if stored_path is None or stored_path == "":
                    unchanged_fields += 1
                    print(f"{doc_id} {field} unchanged")
                    continue
                try:
                    if not isinstance(stored_path, str):
                        raise ValueError("storage path must be a string")
                    normalized_path = normalize_document_storage_path(
                        user_id=user_id,
                        doc_id=doc_id,
                        storage_path=stored_path,
                    )
                except ValueError:
                    rejected_fields += 1
                    print(f"{doc_id} {field} rejected")
                else:
                    if normalized_path == stored_path:
                        unchanged_fields += 1
                        print(f"{doc_id} {field} unchanged")
                    else:
                        changed_fields += 1
                        pending[field] = normalized_path
                        print(f"{doc_id} {field} convertible")

            if apply and pending:
                await update_owned_document_paths(
                    doc_id=doc_id,
                    user_id=user_id,
                    paths=pending,
                )
                applied_fields += len(pending)

        offset += batch_size

    summary = MigrationSummary(
        scanned_rows=scanned_rows,
        changed_fields=changed_fields,
        applied_fields=applied_fields,
        unchanged_fields=unchanged_fields,
        rejected_fields=rejected_fields,
    )
    print(
        "summary "
        f"scanned_rows={summary.scanned_rows} "
        f"changed_fields={summary.changed_fields} "
        f"applied_fields={summary.applied_fields} "
        f"unchanged_fields={summary.unchanged_fields} "
        f"rejected_fields={summary.rejected_fields}"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize legacy document paths; dry-run unless --apply is set."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write only safely convertible document paths",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="rows per request (1-1000; default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the migration CLI without exposing backend error details."""
    args = _parse_args()
    try:
        asyncio.run(migrate_document_paths(apply=args.apply, batch_size=args.batch_size))
    except Exception:
        raise SystemExit("Migration failed") from None


if __name__ == "__main__":
    main()

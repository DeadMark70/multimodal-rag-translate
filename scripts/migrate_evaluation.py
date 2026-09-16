"""Initialize PostgreSQL and optionally import an offline SQLite evaluation DB.

Set EVALUATION_DATABASE_URL. Stop evaluation writers and back up SQLite first.
The source is opened read-only; an import requires empty destination tables.
All imported rows and their checksums are verified in the same transaction.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.postgres import close_db, connect_db, force_init_db  # noqa: E402


def row_bytes(row: tuple[Any, ...]) -> bytes:
    """Normalize numeric DB representations without changing strings or NULL."""
    values = [
        ["number", str(value) if isinstance(value, int) else format(value, ".17g")]
        if isinstance(value, (int, float)) else ["value", value]
        for value in row
    ]
    return (json.dumps(values, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def sqlite_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


async def import_sqlite(source: Path) -> dict[str, Any]:
    """Copy a consistent read-only snapshot and reject nonempty destinations."""
    if not source.is_file():
        raise ValueError("SQLite source does not exist")
    sqlite = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    report: dict[str, Any] = {}
    try:
        sqlite.execute("PRAGMA query_only=ON")
        sqlite.execute("BEGIN")
        source_tables = [r[0] for r in sqlite.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        )]
        async with connect_db() as target:
            await target.execute("SELECT pg_advisory_xact_lock(74503101)")
            target_rows = await (await target.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = current_schema() ORDER BY ordinal_position"
            )).fetchall()
            columns_by_table: dict[str, set[str]] = {}
            types_by_column = {}
            for row in target_rows:
                columns_by_table.setdefault(row["table_name"], set()).add(row["column_name"])
                types_by_column[row["table_name"], row["column_name"]] = row["data_type"]
            unknown = set(source_tables) - columns_by_table.keys()
            if unknown:
                raise ValueError(f"Unknown source tables: {sorted(unknown)}")
            foreign_keys = await (await target.execute(
                "SELECT child.relname AS child, parent.relname AS parent "
                "FROM pg_constraint fk JOIN pg_class child ON child.oid=fk.conrelid "
                "JOIN pg_class parent ON parent.oid=fk.confrelid "
                "JOIN pg_namespace ns ON ns.oid=child.relnamespace "
                "WHERE fk.contype='f' AND ns.nspname=current_schema()"
            )).fetchall()
            dependencies = {table: set() for table in source_tables}
            for fk in foreign_keys:
                if fk["child"] in dependencies and fk["parent"] in dependencies:
                    dependencies[fk["child"]].add(fk["parent"])
            ordered: list[str] = []
            while dependencies:
                ready = [table for table, parents in dependencies.items() if not parents]
                if not ready:
                    raise ValueError("Cyclic table dependencies require an explicit migration")
                ordered.extend(ready)
                for table in ready:
                    del dependencies[table]
                for parents in dependencies.values():
                    parents.difference_update(ready)
            source_tables = ordered
            # Check every original table before inserting the first row.
            for table in source_tables:
                row = await (await target.execute(sql.SQL(
                    "SELECT EXISTS(SELECT 1 FROM {}) AS populated"
                ).format(sql.Identifier(table)))).fetchone()
                if row["populated"]:
                    raise ValueError(f"Destination table is not empty: {table}")
            for table in source_tables:
                info = sqlite.execute(f"PRAGMA table_info({sqlite_identifier(table)})").fetchall()
                columns = [row[1] for row in info]
                unknown_columns = set(columns) - columns_by_table[table]
                if unknown_columns:
                    raise ValueError(f"Unknown source columns in {table}: {sorted(unknown_columns)}")
                keys = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
                if not keys:
                    raise ValueError(f"Source table has no primary key: {table}")
                projection = ", ".join(map(sqlite_identifier, columns))
                order = ", ".join(map(sqlite_identifier, keys))
                rows = sqlite.execute(
                    f"SELECT {projection} FROM {sqlite_identifier(table)} ORDER BY {order}"
                )
                source_hash = hashlib.sha256()
                count = 0
                async with target.cursor() as cursor:
                    async with cursor.copy(sql.SQL("COPY {} ({}) FROM STDIN").format(
                        sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, columns))
                    )) as copy:
                        for row in rows:
                            await copy.write_row(row)
                            source_hash.update(row_bytes(row))
                            count += 1
                target_hash = hashlib.sha256()
                target_count = 0
                async with target.cursor(name="verify_import") as cursor:
                    await cursor.execute(sql.SQL("SELECT {} FROM {} ORDER BY {}").format(
                        sql.SQL(", ").join(map(sql.Identifier, columns)),
                        sql.Identifier(table),
                        sql.SQL(", ").join(
                            sql.SQL('{} COLLATE "C"').format(sql.Identifier(key))
                            if types_by_column[table, key] == "text" else sql.Identifier(key)
                            for key in keys
                        ),
                    ))
                    async for row in cursor:
                        target_hash.update(row_bytes(tuple(row[c] for c in columns)))
                        target_count += 1
                if count != target_count or source_hash.digest() != target_hash.digest():
                    raise ValueError(f"Imported content verification failed: {table}")
                report[table] = {"rows": count, "sha256": source_hash.hexdigest()}
            # Foreign keys and unique constraints have been active throughout COPY.
        async with connect_db() as target:
            await target.execute("ANALYZE")
    finally:
        sqlite.close()
    return {"verified": True, "tables": report}


async def warm_summaries() -> dict[str, Any]:
    """Build page caches from stored evidence only, without running a worker."""
    from evaluation.analysis_cache import build_analysis
    from evaluation.research_analytics import ResearchAnalyticsService

    service = ResearchAnalyticsService()
    kinds = (
        ("summary", service.get_summary),
        ("questions", service.get_question_comparison),
        ("behavior", service.get_agent_behavior),
    )
    async with connect_db() as connection:
        campaigns = await (await connection.execute("SELECT id,user_id FROM campaigns ORDER BY id")).fetchall()
    count = 0
    for campaign in campaigns:
        for kind, method in kinds:
            await build_analysis(
                user_id=campaign["user_id"], campaign_id=campaign["id"], kind=kind,
                loader=lambda: method(user_id=campaign["user_id"], campaign_id=campaign["id"]),
            )
            count += 1
    return {"pages": count, "campaigns": len(campaigns)}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, help="Offline/read-only SQLite source")
    parser.add_argument("--report", type=Path, help="Write row counts and checksums as JSON")
    parser.add_argument("--warm-summaries", action="store_true", help="Build stored analysis pages without model calls")
    args = parser.parse_args()
    try:
        await force_init_db()
        report = await import_sqlite(args.sqlite) if args.sqlite else {"schema_ready": True}
        if args.warm_summaries:
            report["warmup"] = await warm_summaries()
        content = json.dumps(report, ensure_ascii=False, indent=2)
        if args.report:
            args.report.write_text(content + "\n", encoding="utf-8")
        print(content)
    finally:
        await close_db()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

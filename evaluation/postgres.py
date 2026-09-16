"""PostgreSQL pool and explicit evaluation schema migrations."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

MIGRATIONS = Path(__file__).with_name("migrations")
_pool: AsyncConnectionPool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None
_checked = False


class RepositoryConnection:
    """Bind repository positional parameters to a PostgreSQL connection."""

    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def execute(self, query: str, parameters: tuple | list | None = None):
        if parameters is not None:
            query = query.replace("%", "%%").replace("?", "%s")
        return await self.connection.execute(query, parameters)

    async def commit(self) -> None:
        await self.connection.commit()

    async def rollback(self) -> None:
        await self.connection.rollback()


def database_url() -> str:
    value = os.getenv("EVALUATION_DATABASE_URL", "")
    if not value:
        raise RuntimeError("Set EVALUATION_DATABASE_URL and run scripts/migrate_evaluation.py first")
    return value


async def get_pool() -> AsyncConnectionPool:
    global _pool, _pool_loop, _checked
    loop = asyncio.get_running_loop()
    if _pool is None or _pool_loop is not loop:
        _checked = False
        _pool_loop = loop
        _pool = AsyncConnectionPool(
            database_url(), open=False, min_size=1,
            max_size=int(os.getenv("EVALUATION_DB_POOL_SIZE", "10")),
            kwargs={"row_factory": dict_row},
        )
        await _pool.open(wait=True)
    return _pool


@asynccontextmanager
async def connect_db() -> AsyncIterator[AsyncConnection]:
    pool = await get_pool()
    async with pool.connection() as connection:
        yield connection


async def close_db() -> None:
    global _pool, _pool_loop, _checked
    if _pool is not None:
        await _pool.close()
    _pool = None
    _pool_loop = None
    _checked = False


async def init_db() -> None:
    """Validate once per pool. Requests do not perform schema changes."""
    global _checked
    await get_pool()
    if _checked:
        return
    async with connect_db() as connection:
        row = await (await connection.execute(
            "SELECT to_regclass('evaluation_schema_migrations') AS name"
        )).fetchone()
        if not row or row["name"] is None:
            raise RuntimeError("Evaluation schema missing; run scripts/migrate_evaluation.py")
        rows = await (await connection.execute(
            "SELECT version FROM evaluation_schema_migrations"
        )).fetchall()
    if {p.name for p in MIGRATIONS.glob("*.sql")} != {r["version"] for r in rows}:
        raise RuntimeError("Evaluation schema mismatch; run scripts/migrate_evaluation.py")
    _checked = True


async def force_init_db() -> None:
    """Apply versioned SQL from deployment tooling, never an API request."""
    global _checked
    async with connect_db() as connection:
        await connection.execute("SELECT pg_advisory_xact_lock(74503101)")
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS evaluation_schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        rows = await (await connection.execute(
            "SELECT version FROM evaluation_schema_migrations"
        )).fetchall()
        applied = {row["version"] for row in rows}
        for path in sorted(MIGRATIONS.glob("*.sql")):
            if path.name in applied:
                continue
            await connection.execute(path.read_text(encoding="utf-8"), prepare=False)
            await connection.execute(
                "INSERT INTO evaluation_schema_migrations(version) VALUES (%s)", (path.name,)
            )
    _checked = True

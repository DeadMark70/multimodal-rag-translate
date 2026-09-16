"""Run repository contracts against isolated PostgreSQL schemas."""

import asyncio
import os
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop_policy():
    if os.name == "nt":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture(autouse=True)
async def postgres_database(monkeypatch):
    dsn = os.getenv("EVALUATION_TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("Set EVALUATION_TEST_POSTGRES_URL to an isolated PostgreSQL instance")
    from evaluation.postgres import close_db, force_init_db

    schema = "test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    monkeypatch.setenv("EVALUATION_DATABASE_URL", make_conninfo(dsn, options=f"-csearch_path={schema}"))
    try:
        await force_init_db()
        yield
    finally:
        await close_db()
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))

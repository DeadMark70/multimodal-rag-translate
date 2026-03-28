import os

import pytest
from supabase_client import supabase


def test_conversations_has_metadata_column():
    """
    Test that the conversations table has a metadata column.

    This test requires real Supabase connectivity. In CI network-blocked mode,
    it is intentionally skipped because the environment forbids external calls.
    """
    if os.getenv("CI_BLOCK_EXTERNAL_NETWORK", "false").strip().lower() in {"1", "true", "yes", "on"}:
        pytest.skip("External Supabase access is blocked in CI network-guard mode.")

    try:
        response = supabase.table("conversations").select("metadata").limit(1).execute()
        assert "metadata" in response.data[0] if response.data else True
    except Exception as e:
        if "column conversations.metadata does not exist" in str(e).lower() or "not found" in str(e).lower():
            pytest.fail(f"Metadata column missing in conversations table: {e}")
        else:
            pytest.fail(f"Could not select metadata column: {e}")


if __name__ == "__main__":
    test_conversations_has_metadata_column()

import pytest
from supabase_client import supabase

def test_conversations_has_metadata_column():
    """
    Test that the conversations table has a metadata column.
    This test is expected to fail before the migration is applied.
    """
    # Fetch one row (or just the structure) from conversations
    # We can use a RPC or just a simple select
    try:
        response = supabase.table("conversations").select("metadata").limit(1).execute()
        # If we can select 'metadata', it means the column exists
        assert "metadata" in response.data[0] if response.data else True
    except Exception as e:
        # If the column doesn't exist, Supabase/Postgrest usually returns an error
        if "column conversations.metadata does not exist" in str(e).lower() or "not found" in str(e).lower():
            pytest.fail(f"Metadata column missing in conversations table: {e}")
        else:
            # Other errors might happen, but if it's specifically about the column missing, we fail.
            # In some cases, selecting a non-existent column returns a 400 error.
            pytest.fail(f"Could not select metadata column: {e}")

if __name__ == "__main__":
    # For manual running
    test_conversations_has_metadata_column()

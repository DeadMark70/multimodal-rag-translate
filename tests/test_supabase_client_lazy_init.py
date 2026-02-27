"""Tests for lazy Supabase client initialization behavior."""

from unittest.mock import MagicMock, patch

from supabase_client import get_supabase, reset_supabase_for_tests, supabase


def test_supabase_client_is_lazy(monkeypatch) -> None:
    """create_client should not run until the client is first accessed."""
    reset_supabase_for_tests()
    monkeypatch.setenv("SUPABASE_URL", "https://demo.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "demo-key")

    with patch("supabase_client.create_client", return_value=MagicMock()) as mock_create:
        assert mock_create.call_count == 0

        client = get_supabase()
        assert client is not None
        assert mock_create.call_count == 1

        # Cached client should be reused without re-initializing.
        _ = get_supabase()
        assert mock_create.call_count == 1


def test_supabase_client_handles_missing_credentials(monkeypatch) -> None:
    """Client should remain unavailable when required env vars are absent."""
    reset_supabase_for_tests()
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    with patch("supabase_client.create_client") as mock_create:
        assert get_supabase() is None
        assert not supabase
        assert mock_create.call_count == 0

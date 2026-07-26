"""Integration tests: real OpenAI embeddings + real Postgres+pgvector.

These tests make real API calls (costs money) and require a running Postgres
instance. They are marked ``@pytest.mark.integration`` and excluded from
default CI runs. Enable via ``RUN_LIVE_API_TESTS=1``.
"""

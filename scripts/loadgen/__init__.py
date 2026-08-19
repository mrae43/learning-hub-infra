"""Locust load generator for Learning Hub (decision #271, issue #290).

Black-box HTTP traffic source for load runs against the deployed api. Drives
``/query`` (~75%) and ``/dive`` (~25%) as weighted user scenarios plus a
low-rate ``/health`` liveness task. Two run profiles — ``volume`` (mock
upstream, sustained) and ``smoke`` (real API, budgeted) — are selected by the
``LOAD_PROFILE`` env var. Imports nothing from ``core``/``retrieval_qa``/
``depth_dive``/``api``, so the import-linter contracts are untouched.
"""

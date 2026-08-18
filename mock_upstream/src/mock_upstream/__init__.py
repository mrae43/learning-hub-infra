"""The mock upstream package: an OpenAI-compatible stand-in for load runs.

Serves deterministic, shape-valid OpenAI embeddings, chat-completions, and
Responses (web-search) responses so volume load runs exercise the deployed
service without spending real API budget or hitting rate limits. See
``CONTEXT.md`` "Mock Upstream" and wayfinder ticket #265 for the design
decisions.
"""

__all__: list[str] = []

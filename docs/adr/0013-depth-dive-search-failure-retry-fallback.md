# 0013 — Depth Dive Web Search Failure Handling: Retry Once, Fallback With a Note

## Status
Accepted

## Context
ADR-0012 established that Depth Dive's web search is agentic — the model judges per-request whether external material would strengthen a Depth Dive, and calls search only when it judges yes. This left open what happens when that judgment is made but the search call itself fails (timeout, API error, no useful results). Per Step 11 of the project blueprint (recover instead of stopping blindly), a bare failure shouldn't block the Depth Dive from being produced at all.

Two things needed deciding: whether to retry before giving up, and whether the user should know a search was attempted and failed versus receiving a passage-only Depth Dive with no indication anything was missing.

## Decision
- **Retry:** one bounded retry on search failure before falling back — most search failures are plausibly transient (network blip, rate limit), and a single retry is cheap relative to the value of getting the intended grounding.
- **Fallback:** if the retry also fails, the Depth Dive still renders (passage-only, no external material), but includes a brief note that an external reference was sought and unavailable — not a silent fallback. This matters specifically because Depth Dive's value proposition depends on the user trusting the grounding level of what they're reading; a silently-degraded Depth Dive risks the user assuming "this is as good as it gets" when the system actually judged more grounding was warranted and couldn't deliver it.

## Consequences
- Depth Dive's response needs a field or mechanism to carry this "search was attempted and failed" signal through to the rendered output — likely part of Depth Dive's eventual structured response schema (parallel to Retrieval QA's `grounded` field), left as an implementation detail for when that schema is designed.
- One retry adds latency on the failure path only (success path is unaffected) — acceptable given search failures should be the exception, not the common case.
- This is consistent with Retrieval QA's `grounded: false` pattern: both modules prefer telling the user "this is a lower-confidence result" over silently presenting a degraded response as if it were the intended one.
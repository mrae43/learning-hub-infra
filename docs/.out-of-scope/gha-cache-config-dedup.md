# Docker cache config deduplication

This project does not extract the GitHub Actions cache configuration (`cache-from` / `cache-to` with `type=gha`) into a shared composite action across the PR and release workflows.

## Why this is out of scope

The PR pipeline's `docker-build` job and the release workflow's image build use the same two-line cache pair:

```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

Deduplicating this into a composite action is the textbook fix for the duplication, but at this repo's scale it is over-engineering:

- The duplication is only two lines, and keeping it inline makes the two pipelines' cache contract visibly identical, which reduces the risk of drift between them.
- The spec for the pre-merge build gate (issue #205) already rejected PR-scoped cache isolation as unnecessary at this traffic level; a shared composite action is a similar kind of speculative abstraction.
- It aligns with the repo's explicit-over-implicit preference in the coding standards.

## Prior requests

- #211 — "Deduplicate GHA cache config between PR docker-build and release workflows"

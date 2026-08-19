# 0023 — Claude Code Plugin as the MVP Client Interface

## Status

Accepted

## Context

`ADR-0020` assigned the Depth Dive client to "a separate UI repo" that owns learner state, with the backend exposing a contract the UI could mock against. That UI repo has not been built, and building a full web client is deferred. Depth Dive now ships as self-contained HTML (`ADR-0022`), which needs only a browser to render — it does not need a standing web frontend.

This raises the question of how the learner reaches the artifact. A Claude Code plugin (commands + skills + an MCP server, per `to-plugins.md`) can act as the interface shell: the coding agent invokes the backend, receives the self-contained HTML, and hands the learner a link to open. This defers the web UI without deferring a usable interface.

## Decision

1. **A Claude Code plugin is the MVP client interface for Depth Dive.** It is the shell the learner talks to; it is not part of the backend and does not replace it.

2. **The plugin is a thin stateless layer on the existing backend.** The FastAPI/OpenAPI surface (`POST /dive`) remains the source of truth; the plugin's MCP server is a stateless RPC bridge that exposes `/dive` and returns the rendered self-contained HTML (`ADR-0022`). Scope is `depth_dive` only — the plugin does not wrap Retrieval QA, ingestion, or document status for MVP.

3. **Delivery is ephemeral.** The MCP tool returns the HTML; the agent writes it to a temporary directory and hands the learner a `file://` link. Nothing is persisted server-side; there is no serving surface. This preserves the `ADR-0020` statelessness boundary (the artifact is built per request; it does not accumulate across requests).

4. **The plugin replaces `ADR-0020`'s "separate UI repo" for MVP.** Learner state remains unowned and deferred — the plugin is stateless, consistent with Depth Dive's no-per-learner-state boundary.

5. **Plugin layout** follows `to-plugins.md` §4: `.claude-plugin/plugin.json` (metadata), a `commands/animate.md` slash command as the explicit entry point, an `html-animation` skill holding the self-contained-HTML output constraints (extracted from the `mattpocock/skills` `/teach` pattern, minus its session-state logic), and `.mcp.json` pointing at the backend.

## Consequences

- `ADR-0020`'s "Owner of learner state: a separate UI repo" is amended: the plugin is the MVP client shell; learner state remains deferred (no in-repo owner). The eventual web UI can still be built later against the unchanged `/dive` contract — the plugin does not freeze it out.
- No new HTTP endpoint or serving surface is introduced. The MCP server exposes existing backend capability, not a new API; the HTML is produced server-side (`ADR-0022`) so self-containment is enforced in code, never trusted to the agent or the prompt.
- The plugin introduces a distinct artifact class to the repo (a `plugin.json`, a command, a skill, and an `.mcp.json` alongside the Python workspace). These are distribution/packaging concerns, not a new backend package; the monorepo's import-boundary rules (`ADR-0011`) are unaffected.
- The skill must stay auditable against `to-plugins.md` §3: no reference to persistent local state files (the `/teach`-style `RESOURCES.md` pattern would reintroduce filesystem state — excluded here).

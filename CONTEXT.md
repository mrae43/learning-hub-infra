# CONTEXT.md
> Glossary of domain terms for the Learning Hub project. No implementation details — terms and their meaning only.

## Retrieval QA
The closed-book question-answering system. Scope is strictly bounded to the model plus the user's ingested documents — no web search, no external tools. Output format is plain text. This is Retrieval QA's entire job; it does not produce artifacts, code samples, or cite external sources beyond the ingested corpus.

## Depth Dive
The synthesis system that consumes Retrieval QA's retrieval layer and produces Depth Dives — richer, non-plain-text outputs. Unlike Retrieval QA, Depth Dive is permitted to extend beyond the ingested corpus via web search. Depth Dive ships after Retrieval QA is proven, since it depends on Retrieval QA's retrieval quality.

## Document-Type Chunker
A structure-aware chunking strategy specific to each ingested document type, rather than one fixed-size splitter applied uniformly. Papers chunk along section/subsection boundaries, books along chapter/heading boundaries, documentation along page/API-entry boundaries. Chosen as a separation-of-concerns principle for the hand-rolled pipeline (see ADR-0003) — each document type owns its own chunking logic rather than sharing a generic splitter.

## Ingested Corpus
The full set of user-uploaded papers, books, and documentation, stored in a single **global** vector database (not siloed per document or per session). Retrieval QA retrieves across the entire ingested corpus by default — cross-document QA (e.g. "compare how paper X and paper Y define attention") is in scope, not a special mode.

## Injected Context
The specific chunks retrieved for a given query and placed into the model's prompt at inference time. A per-query subset of the ingested corpus — not the whole corpus, and never anything from outside it (no web search) for Retrieval QA.

## Captured Passage
The specific paragraph(s), figure, or excerpt a user selects from an ingested document while reading, to anchor a Depth Dive request. Example: mid-way through Chapter 1 of a book, the user captures a paragraph they don't understand and hands it to Depth Dive as the grounding context for that request. Distinct from "injected context" — a captured passage is user-selected and explicit, not retrieved.

## Cross-Reference (implicit)
A property of Retrieval QA's global-corpus retrieval, not a separate feature: because all ingested documents share one vector database, a query naturally surfaces injected context from any related document, regardless of which document the user is "currently" reading. In MVP by default — no extra machinery required beyond global retrieval.

## Concept Linking (post-MVP)
The explicit, query-independent capability of proactively surfacing relationships between documents (e.g. "this passage relates to a paper you uploaded last week") without the user asking. Requires its own architecture (entity/concept extraction, a relation layer over the vector store, a UI surface) and its own evaluation criteria. Deferred until Retrieval QA and B are proven.

## Depth Dive
Depth Dive's MVP output format: a **dual-coding** explanation of a captured passage — text paired with a diagram, carousel, or coding example. Stateless; no quizzing, no scheduling. Final term — chosen to avoid colliding with "Artifacts" as a pre-existing product term, and to capture both the depth of understanding it provides and the fact that it's triggered by diving into a specific captured passage.

## Retrieval Practice / Spaced Repetition (post-MVP)
Testing-effect and scheduling-based learning mechanisms (quizzes on captured passages, scheduled resurfacing of concepts) considered part of the "neuroscience-backed" goal but explicitly out of MVP. Requires durable per-concept state (long-term memory), unlike dual coding which is stateless per-request.
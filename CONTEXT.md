# CONTEXT.md
> Glossary of domain terms for the Learning Hub project. No implementation details — terms and their meaning only.

## Retrieval QA
The closed-book question-answering system. Scope is strictly bounded to the model plus the user's ingested documents — no web search, no external tools. Output format is plain text. This is Retrieval QA's entire job; it does not produce artifacts, code samples, or cite external sources beyond the ingested corpus.

## Depth Dive
The synthesis system that consumes Retrieval QA's retrieval layer and produces richer, non-plain-text outputs called Depth Dives. MVP output format is a **dual-coding** explanation of a captured passage — text paired with a diagram, carousel, or coding example. Stateless; no quizzing, no scheduling. Unlike Retrieval QA, Depth Dive is permitted to extend beyond the ingested corpus via agentic web search. Ships after Retrieval QA is proven, since it depends on Retrieval QA's retrieval quality. Term chosen to avoid collision with "Artifacts" as a pre-existing product term, and to capture both the depth of understanding it provides and the fact that it's triggered by diving into a specific captured passage.

## Document-Type Chunker
A structure-aware chunking strategy specific to each ingested document type, rather than one fixed-size splitter applied uniformly. Papers chunk along section/subsection boundaries, books along chapter/heading boundaries, documentation along page/API-entry boundaries. Chosen as a separation-of-concerns principle for the hand-rolled pipeline (see ADR-0003) — each document type owns its own chunking logic rather than sharing a generic splitter.

## Chunk
The atomic retrievable unit of an ingested document, produced by a Document-Type Chunker. Has stable identity, an ordered position within its source document, and content; once embedded, it becomes retrievable as Injected Context and citable in a Retrieval QA response. Distinct from a "Captured Passage" — a captured passage is user-selected and grounds Depth Dive; a chunk is system-produced and grounds Retrieval QA.

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

## Retrieval Practice / Spaced Repetition (post-MVP)
Testing-effect and scheduling-based learning mechanisms (quizzes on captured passages, scheduled resurfacing of concepts) considered part of the "neuroscience-backed" goal but explicitly out of MVP. Requires durable per-concept state (long-term memory), unlike dual coding which is stateless per-request.

## Parent Chunk
The enclosing structural unit (section, chapter, API page) produced by a Document-Type Chunker. Not embedded directly. Contains one or more Child Chunks. At retrieval time, the parent replaces matched child chunks before being handed to the LLM for generation.

## Child Chunk
A fixed-size (~512 tokens, 15% overlap) recursive split of a Parent Chunk. Embedded and indexed for retrieval (both dense pgvector and sparse tsvector). A child points to its parent via `parent_chunk_id`. Only children are matched by the query; the parent is what reaches generation.

## Hybrid Search
A retrieval strategy combining dense (embedding-based pgvector cosine search) and sparse (PostgreSQL `tsvector` full-text search) passes, fused via Reciprocal Rank Fusion. Recovers exact-match queries (function names, API endpoints, error codes, symbols) that pure dense retrieval misses.

## Reranking
A second-pass relevance filter that scores the top-K candidates from hybrid search using a cross-encoder (Cohere Rerank for prototyping, planned swap to local BGE-reranker). Keeps the top-5 for generation. Implemented after hybrid search per the RAG reference guide's priority order.

## Query Decomposition (post-MVP)
The technique of splitting a complex multi-hop question into simpler sub-queries, retrieving for each, then synthesizing the results. Deferred until parent-child chunking, hybrid search, and reranking are implemented and evaluated.

## Evaluation

**Content-Signature Labeling**:
A ground-truth labeling strategy for retrieval evaluation where expected passages are identified by a distinctive substring of their text (or a SHA-256 of that substring). Invariant to chunk boundaries — unlike position ranges or page numbers — so the same labeled queries work across any chunk-size configuration without re-labeling.

**Eval Query Taxonomy**:
Four-stratum classification of eval queries:
- *Concept lookup* (dense-friendly) — single-fact lookup ("What is attention?")
- *Exact-match / keyword* (sparse-friendly) — API names, error codes, CLI flags
- *Context-dependent* (parent-child matters) — requires the enclosing section to answer precisely
- *Multi-hop / reasoning* (decomposition prep) — relates two or more concepts across different parts of the corpus
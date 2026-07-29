# RAG System Reference Guide

## Overview

Retrieval-Augmented Generation (RAG) combines a retriever and a generator to produce grounded answers. The retriever fetches relevant passages from a knowledge corpus, and the generator produces an answer conditioned on those passages.

## 1. Ingestion Pipeline

### Chunking Strategies

The ingestion pipeline converts source documents into retrievable chunks. Different document types require different chunking strategies.

#### Recursive Character Splitter
Splits text on separators in order of priority: paragraph breaks, sentence boundaries, word boundaries. This is the simplest strategy and works as a fallback for any text.

#### Semantic Chunking
Uses embedding similarity to detect natural topic boundaries. When the embedding distance between consecutive sentences exceeds a threshold, a new chunk is started. This preserves semantic coherence but is more expensive than recursive splitting.

#### Document-Type Chunkers
Structure-aware chunkers that exploit document structure:
- Paper chunker: splits along IMRaD section boundaries (Introduction, Methods, Results, Discussion)
- Book chunker: splits along chapter and heading boundaries
- Documentation chunker: splits along heading hierarchy (h1-h3), preserves code blocks intact

### Metadata

Every chunk should carry metadata: source document ID, title, section heading, position, document type, and timestamp. Metadata enables filtering at retrieval time and helps the LLM cite sources correctly.

## 2. Retrieval

### Embedding-Based Retrieval

Documents are encoded into dense vector embeddings using models like text-embedding-3-small or text-embedding-004. Queries are embedded using the same model. Retrieval finds the nearest neighbors in embedding space using cosine similarity.

### Sparse Retrieval (BM25)

BM25 is a bag-of-words retrieval method that scores documents based on term frequency and inverse document frequency. It excels at exact-match queries where the query contains rare or distinctive terms.

### Hybrid Search

Hybrid search combines dense (embedding) and sparse (BM25) retrieval via Reciprocal Rank Fusion (RRF). RRF combines the rank positions from each method:
RRF_score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d))
where k is a constant (typically 60). Hybrid search recovers exact-match queries that pure dense retrieval misses.

### Reranking

After initial retrieval, a cross-encoder reranker scores the candidate passages. Unlike bi-encoders (which produce independent embeddings), cross-encoders process query-passage pairs jointly, producing more accurate relevance scores. The top-K candidates from hybrid search (typically 20-50) are reranked, and the top 3-10 are kept for generation.

### Parent-Child Retrieval

In parent-child chunking, small child chunks (~512 tokens) are embedded and matched by the query. The parent chunk (the enclosing section) replaces the matched children before being handed to the LLM. This resolves the precision-vs-context tradeoff.

## 3. Query Processing

### Query Rewriting

Ambiguous or conversational queries are rewritten into standalone, retrieval-friendly forms. This is particularly useful in multi-turn settings.

### Query Decomposition

Complex multi-hop questions are split into simpler sub-queries. Each sub-query is retrieved independently, and results are merged before generation. Decomposition should be gated — only triggered when the query is detected as multi-part.

## 4. Evaluation

### Retrieval Metrics

Recall@k measures the proportion of relevant documents retrieved in the top-k results. Mean Reciprocal Rank (MRR) measures the average rank position of the first relevant document.

### Eval Query Taxonomy

Queries are categorized into four strata for systematic evaluation:
- Concept lookup (dense-friendly): single-fact lookup
- Exact match / keyword (sparse-friendly): API names, error codes, CLI flags
- Context-dependent (parent-child matters): requires the enclosing section
- Multi-hop / reasoning (decomposition prep): relates two or more concepts

### Content-Signature Labeling

Expected passages are identified by a distinctive substring of their text. This labeling is invariant to chunk boundaries, so the same labeled queries work across any chunk-size configuration without re-labeling.

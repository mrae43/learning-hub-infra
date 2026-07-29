# vLLM: Efficient LLM Serving with PagedAttention

## Abstract

vLLM is a high-throughput LLM serving system that introduces PagedAttention, a novel attention algorithm inspired by virtual memory and paging in operating systems. PagedAttention manages the KV cache in fixed-size blocks (pages), enabling efficient memory sharing and reducing memory waste from fragmentation.

## 1. Introduction

Serving large language models is memory-bound because the KV cache grows with both batch size and sequence length. The standard approach to KV cache management suffers from memory fragmentation and can only accommodate limited memory sharing across sequences. PagedAttention addresses both problems by managing the KV cache as a paged system.

## 2. PagedAttention

### Block-Based KV Cache

In PagedAttention, the KV cache for each sequence is divided into blocks of fixed size (e.g., 16 tokens per block). Each block maps to a physical block in the global block table. This is analogous to how virtual memory pages map to physical memory frames. The block table maintains the mapping from logical blocks to physical blocks.

### Key Operations

PagedAttention operates on blocks rather than individual tokens. When computing attention, the system looks up the physical blocks for the required KV cache entries using the block table. This block-level operation allows efficient memory management because the system allocates memory at block granularity.

## 3. Memory Management

### Fragmentation Elimination

Standard KV cache pre-allocates contiguous memory for the maximum sequence length, leading to internal fragmentation when sequences are shorter than the maximum. PagedAttention eliminates this by allocating blocks on demand, only when tokens are generated. External fragmentation is handled by the block-level allocation.

### Memory Sharing

PagedAttention enables efficient memory sharing across sequences. When multiple sequences share the same prefix (as in beam search or parallel sampling), their initial KV cache blocks can be shared rather than duplicated. The block table supports copy-on-write, where shared blocks are marked as read-only until a sequence writes new KV entries, at which point a new physical block is allocated.

### Copy-on-Write Mechanism

The copy-on-write mechanism works as follows: when a block is shared across multiple sequences, it is designated as read-only. When a sequence needs to write to a shared block, the system allocates a new physical block, copies the old data, and updates the sequence's block table entry. The old block remains available for other sequences.

## 4. Scheduling

### Iteration-Level Scheduling

vLLM uses iteration-level scheduling, where the scheduler decides which sequences to process at each iteration. This is different from request-level scheduling used in traditional systems. Iteration-level scheduling allows the system to dynamically adjust the batch composition based on current memory availability.

### Memory-Aware Scheduling

The scheduler tracks available physical blocks and only admits sequences when sufficient physical blocks are available. This prevents out-of-memory errors and allows the system to maximize throughput by filling available memory.

## 5. System Architecture

The vLLM system consists of a scheduler, a model executor, and a memory manager. The scheduler manages the iteration-level scheduling decisions. The model executor runs the actual model inference. The memory manager handles block allocation, deallocation, and copy-on-write operations.

## 6. Evaluation

vLLM achieves up to 2-4x higher throughput compared to FasterTransformer and Orca. The throughput gain comes primarily from the PagedAttention memory management, which reduces memory waste and enables larger batch sizes. The system also achieves near-linear scaling with the number of GPUs.

## 7. Future Directions

Future work includes extending PagedAttention to multi-GPU settings with distributed block tables, supporting more sophisticated eviction policies, and integrating with prefix caching systems.

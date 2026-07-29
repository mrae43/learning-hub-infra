# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## Abstract

FlashAttention is an algorithm that computes exact attention with significantly reduced memory reads and writes. It makes attention faster and more memory-efficient by being aware of the GPU memory hierarchy. The key idea is to tile the attention computation, loading blocks of data from slow HBM to fast SRAM, computing attention over those blocks, and then writing the result back.

## 1. Introduction

The attention mechanism is a fundamental building block of Transformer models. However, the standard attention implementation materializes the N x N attention matrix in HBM (high-bandwidth memory), which has O(N^2) memory complexity. For long sequences, this becomes prohibitive. FlashAttention avoids materializing the full attention matrix by using tiling and recomputation.

## 2. Background: GPU Memory Hierarchy

Modern GPUs have multiple levels of memory. The largest is HBM (high-bandwidth memory, typically 16-80 GB) which has relatively low bandwidth compared to on-chip SRAM. On-chip SRAM (shared memory, typically 192 KB per block on A100) has high bandwidth but very limited capacity. The key to performance is minimizing HBM accesses by maximizing the use of SRAM.

### Memory Hierarchy Details

The A100 GPU has the following memory hierarchy:
- HBM: 40-80 GB, ~1.5-2.0 TB/s bandwidth
- L2 cache: 40 MB
- SRAM/shared memory: 192 KB per streaming multiprocessor (SM), ~19 TB/s aggregate bandwidth

## 3. FlashAttention Algorithm

### Tiling Strategy

FlashAttention tiles the Q, K, and V matrices into blocks that fit in on-chip SRAM. For each block, the algorithm:
1. Loads blocks of Q, K from HBM to SRAM
2. Computes the partial attention scores S = Q * K^T
3. Applies the softmax operation
4. Computes the weighted sum with V
5. Writes the result back to HBM

The tiling is done separately along the query dimension and the key dimension. This ensures that the intermediate N x N attention matrix is never fully materialized in HBM.

### Recomputation

To avoid storing the large attention matrix for the backward pass, FlashAttention recomputes the attention scores during the backward pass using the blocks of Q, K, and V stored in HBM. This recomputation trades off additional FLOPs for reduced memory reads/writes, which is beneficial because SRAM-based computation is much faster than HBM bandwidth.

## 4. Results

FlashAttention achieves up to 2x speedup over the standard PyTorch implementation for BERT-base training. For long sequences, the speedup can be even more dramatic. Memory savings scale linearly with sequence length compared to the quadratic memory of standard attention.

## 5. Block-Sparse FlashAttention

For very long sequences, even FlashAttention's tiled approach can be improved by exploiting sparsity. Block-Sparse FlashAttention extends FlashAttention to handle attention matrices where many blocks are known to be zero (for example, due to local or dilated attention patterns). It uses a block-sparse mask to skip computation and memory access for zero blocks.

### Block-Sparsity Mask

The block-sparsity mask is a binary matrix where each entry indicates whether the corresponding block of the attention matrix should be computed. This mask is typically derived from the attention pattern (e.g., local windows, dilated patterns, or learned sparsity). The mask operates on the block level, not on individual elements, to maintain the efficiency of the tiled approach.

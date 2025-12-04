# PQCache
Note that PQCache, like many other baselines, includes several proposals aimed at improving system efficiency, such as optimized CPU–GPU coordination and related design choices. However, in keeping with the guiding principle of Sparse Attention Hub, we isolate and evaluate only the core algorithmic component that is directly relevant to sparse attention.


## 1. PQcache central idea 
PQCache partitions the KV cache into three regions: initial tokens (handled by SinkMasker), local tokens (handled by LocalMasker), and middle tokens. The initial and local tokens are fully retained, while top-k tokens are predicted and selected from the middle region. To enable this efficiently, the K cache of the middle tokens is PQ-quantized. This quantization is performed once, after the context has been processed using dense attention. The resulting centroids are then reused during generation — in our implementation, we employ these centroids for both query processing and generation.

### Example config in sparse-attention-hub
```
    sparse_attention_config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(
            sink_size=128,
        ),
        LocalMaskerConfig(
            window_size=128,
        ),
        PQCacheConfig(
            heavy_size=0.1,
            pq_group_factor=2,  # pq_sub_dim = head_dim // pq_group_factor (e.g., 128 // 2 = 64)
            pq_bits=6,
            kmeans_iter=10,
            init_offset=128,
            metric="euclidean",
        ),
    ])
```

## 2. Matching the author's code behavior exactly in sparse-attention-hub /  Explanation of differences

In Sparse Attention Hub, our goal is not to exactly replicate the behavior of the original PQCache implementation. Instead, we aim to provide a simple, PQ-based method that follows the scoring algorithm proposed in PQCache.

**Known differences:**

1. Question processing: PQCache evaluates questions using dense attention, whereas we use sparse attention. The detailed setup is described in [evaluation_setup](../general/evaluation_setup.md)
2. K-means iterations: PQCache employs dynamic k-means iterations, while we use a fixed number of iterations for simplicity.
3. Similarity metrics: PQCache supports both inner product and Euclidean distance metrics; we currently support only Euclidean.
4. Author's code uses Sklearn's K-means, where as we use custom implemented batched k- means (algorithmically it should be same with potentially different handling of empty clusters etc.)


There might be more differences.

Some sample results with above config.

|                              | multifieldqa_en | hotpotqa |
|------------------------------|-----------------|----------|
| Our Code (10x sparsty,)      | 53.34           | 53.14    |
| Paper numbers (10x sparsity) | 54.97           | 56.35    |


## 5. Performance (Sanity check for research framework):

32K context, Q=1

🔍 Comparative Analysis:
============================================================
📈 Masker Overhead Analysis:
   - Sparse Attention:     0.867 ms
   - Baseline (no maskers): 0.512 ms
   - Masker overhead:      0.355 ms (69.5%)

============================================================


## References
[1] Zhang, H., Ji, X., Chen, Y., Fu, F., Miao, X., Nie, X., Chen, W. and Cui, B., 2025. Pqcache: Product quantization-based kvcache for long context llm inference. Proceedings of the ACM on Management of Data, 3(3), pp.1-30.
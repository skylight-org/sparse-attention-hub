"""Example: Using CUDA Graphs to eliminate launch overhead.

This approach can reduce overhead from 310 μs to ~150-200 μs.
"""

import torch
from typing import Optional, Tuple


class CUDAGraphIndexer:
    """Wrapper that uses CUDA Graphs to accelerate indexer_next."""
    
    def __init__(self, indexer_fn):
        self.indexer_fn = indexer_fn
        self.graph = None
        self.static_inputs = {}
        self.static_outputs = {}
        self.graph_ready = False
        
    def warmup_and_capture(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weight_list_dtype: torch.dtype,
        sink_size: int,
        window_size: int,
        heavy_size: int,
        pq_group_factor: int,
        pq_bits: int,
        kmeans_iter: int,
        init_offset: int,
        metric: str,
        pq_centroids: torch.Tensor,
        pq_codebook: torch.Tensor,
        pq_ip2l2_phi: Optional[torch.Tensor],
    ):
        """Capture the execution as a CUDA graph (one-time cost)."""
        
        # Warmup runs (required for CUDA graphs)
        print("Warming up...")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            for _ in range(10):
                _ = self.indexer_fn(
                    query, key, weight_list_dtype,
                    sink_size, window_size, heavy_size,
                    pq_group_factor, pq_bits, kmeans_iter,
                    init_offset, metric,
                    pq_centroids, pq_codebook, pq_ip2l2_phi
                )
        torch.cuda.current_stream().wait_stream(s)
        
        # Capture the graph
        print("Capturing CUDA graph...")
        self.graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(self.graph):
            self.static_outputs = self.indexer_fn(
                query, key, weight_list_dtype,
                sink_size, window_size, heavy_size,
                pq_group_factor, pq_bits, kmeans_iter,
                init_offset, metric,
                pq_centroids, pq_codebook, pq_ip2l2_phi
            )
        
        self.graph_ready = True
        print("✅ CUDA graph captured!")
        
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weight_list_dtype: torch.dtype,
        sink_size: int,
        window_size: int,
        heavy_size: int,
        pq_group_factor: int,
        pq_bits: int,
        kmeans_iter: int,
        init_offset: int,
        metric: str,
        pq_centroids: torch.Tensor,
        pq_codebook: torch.Tensor,
        pq_ip2l2_phi: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Execute using captured graph (fast!)."""
        
        if not self.graph_ready:
            self.warmup_and_capture(
                query, key, weight_list_dtype,
                sink_size, window_size, heavy_size,
                pq_group_factor, pq_bits, kmeans_iter,
                init_offset, metric,
                pq_centroids, pq_codebook, pq_ip2l2_phi
            )
        
        # Replay the graph - this is FAST!
        # Single API call instead of 30+ kernel launches
        self.graph.replay()
        
        return self.static_outputs


# Usage example:
"""
from iter_6.optimized_indexer import __indexer_next

# Wrap with CUDA graphs
indexer_with_graphs = CUDAGraphIndexer(__indexer_next)

# First call: capture graph (slow, one-time)
result = indexer_with_graphs(query, key, ...)

# Subsequent calls: replay graph (FAST!)
result = indexer_with_graphs(query, key, ...)  # ~350-400 μs instead of 518 μs!
"""

"""
Expected Performance:
- Without CUDA graphs: 518 μs (208 μs CUDA + 310 μs overhead)
- With CUDA graphs:    ~350-400 μs (208 μs CUDA + 150 μs overhead)
- Savings:             ~120-170 μs (38% overhead reduction)

Limitations:
- Graph must be captured for each unique input shape
- Cannot capture dynamic control flow
- All operations must be CUDA operations (no CPU ops in graph)
- Memory addresses must be static
"""


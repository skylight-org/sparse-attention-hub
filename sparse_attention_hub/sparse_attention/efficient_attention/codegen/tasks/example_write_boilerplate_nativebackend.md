Write a boiler plate pqcache/native.py which has 

PQCacheNativeBackendConfig:
	- which should check that it has configs for maskers: SinkMasker, LocalMasker and PQCache.


PQCacheNativeBackend:

__init__:
    extract config parameters

indexer_first / indexer_next:
    should call the global __indexer_first / __indexer_next  function and should pass all relevant inputs config parameters that were extracted and sparse_meta_data if it is not empty.

have unimplemented global functions __indexer_first and __indexer_next which will be implemented later. 



For indexer_first sparse_meta_data is supposed to be empty, remove sparse_meta_data from input for __indexer_next

For indexer_next get the pq_* fields from sparse_meta_data for this layer_idx (which should be in kwargs) and pass it to __indexer_next


__indexer_next (and __indexer_first ) should return sparsity details i.e. sparse_list, sparse_len, weight_list tensors and also should return updated meta data . in this case pq_* tensors. indexer_next function should update sparse_meta_data before returning only the sparsity details 
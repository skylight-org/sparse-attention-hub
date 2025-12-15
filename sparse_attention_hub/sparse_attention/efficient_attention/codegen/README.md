# Codegen Scripts

This directory contains scripts for testing correctness and profiling indexer functions in efficient attention backends.

## Scripts

### `correctness.py`

Tests correctness of `indexer_first` and `indexer_next` methods between research and native backends.

### `profile.py`

Profiles `indexer_first` and `indexer_next` methods using PyTorch profiler and manual timing measurements.

## Usage Examples

### Correctness Testing

#### Basic correctness test for `indexer_first`:

```bash
python -m sparse_attention_hub.sparse_attention.efficient_attention.codegen.correctness \
  --class1 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend \
  --class2 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend \
  --function indexer_first
```

#### Basic correctness test for `indexer_next`:

```bash
python -m sparse_attention_hub.sparse_attention.efficient_attention.codegen.correctness \
  --class1 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend \
  --class2 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend \
  --function indexer_next
```

#### Test with custom number of iterations:

```bash
python -m sparse_attention_hub.sparse_attention.efficient_attention.codegen.correctness \
  --class1 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend \
  --class2 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend \
  --function indexer_first \
  --num-iterations 20
```

#### Test with custom `__indexer_first` function:

```bash
python -m sparse_attention_hub.sparse_attention.efficient_attention.codegen.correctness \
  --class1 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend \
  --class2 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend \
  --function indexer_first \
  --indexer-first-file /path/to/custom_indexer_first.py
```

#### Test with custom `__indexer_next` function:

```bash
python -m sparse_attention_hub.sparse_attention.efficient_attention.codegen.correctness \
  --class1 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend \
  --class2 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend \
  --function indexer_next \
  --indexer-next-file /path/to/custom_indexer_next.py
```

### Profiling

#### Basic profiling for `indexer-first`:

```bash
python -m sparse_attention_hub.sparse_attention.efficient_attention.codegen.profile \
  --class1 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend \
  --class2 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend \
  --function indexer-first
```

#### Basic profiling for `indexer-next`:

```bash
python -m sparse_attention_hub.sparse_attention.efficient_attention.codegen.profile \
  --class1 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend \
  --class2 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend \
  --function indexer-next
```

#### Profiling with custom parameters:

```bash
python -m sparse_attention_hub.sparse_attention.efficient_attention.codegen.profile \
  --class1 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend \
  --class2 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend \
  --function indexer-first \
  --warmup-runs 10 \
  --profile-runs 3 \
  --timing-runs 100
```

#### Profiling with custom `__indexer_first` function:

```bash
python -m sparse_attention_hub.sparse_attention.efficient_attention.codegen.profile \
  --class1 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend \
  --class2 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend \
  --function indexer-first \
  --indexer-first-file /path/to/custom_indexer_first.py
```

#### Profiling with custom `__indexer_next` function:

```bash
python -m sparse_attention_hub.sparse_attention.efficient_attention.codegen.profile \
  --class1 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend \
  --class2 sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend \
  --function indexer-next \
  --indexer-next-file /path/to/custom_indexer_next.py
```

## Parameters

### Common Parameters

- `--class1`: Full path to research backend class (EfficientAttentionResearchBackend)
  - Example: `sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend`
- `--class2`: Full path to native backend class (EfficientAttention)
  - Example: `sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend`
- `--function`: Function to test/profile
  - For `correctness.py`: `indexer_first` or `indexer_next`
  - For `profile.py`: `indexer-first` or `indexer-next`
- `--indexer-first-file`: Optional path to Python file containing `__indexer_first` function to replace
- `--indexer-next-file`: Optional path to Python file containing `__indexer_next` function to replace

### Correctness Script Specific

- `--num-iterations`: Number of test iterations (default: 10)

### Profile Script Specific

- `--warmup-runs`: Number of warmup runs before profiling (default: 5)
- `--profile-runs`: Number of runs to profile with PyTorch profiler (default: 1)
- `--timing-runs`: Number of runs for manual timing measurements (default: 50)

## Output

### Correctness Script

- Prints success/failure status for each iteration
- Exits with code 0 on success, 1 on failure

### Profile Script

- Prints profiling statistics (CPU/CUDA time, memory usage)
- Generates Chrome trace file (`profile_indexer_first_trace.json` or `profile_indexer_next_trace.json`)
- Prints timing statistics (average, median, min, max, std dev)
- Trace files can be viewed at https://ui.perfetto.dev/

## Notes

- Both scripts use sample data generated by `class1.create_sample_data_first()` or `class1.create_sample_data_next()` with parameters: B=1, H=32, num_keys=256, d=128
- The scripts automatically detect and use CUDA if available, otherwise fall back to CPU
- Custom indexer functions must be named `__indexer_first` or `__indexer_next` in the provided file


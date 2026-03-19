## Ray Tune Benchmark Suite

Distributed benchmark suite for sparse attention configurations using Ray.

### 1. Quick Start (Run existing builders on new models / settings / objectives)

- **Optimize configs**

  1. Edit `benchmark/raytune/OPTIMIZATION_EXPERIMENT.py` to choose:
     - **Models**: `MODEL_CONFIGS`, `MODELS`
     - **Tasks**: `TASKS`
     - **Objectives**: `SPARSITY_OBJECTIVES`, `MEMORY_OBJECTIVES`
     - **Builders**: `BUILDER_NAMES`
     - **Search/runtime**: samples, timeouts, context limits, output dirs
  2. Run the optimization:

```bash
python3 benchmark/raytune/run_optimize_configs.py
```

  This writes one JSON config per (model, task, builder, objective) into the configured optimal-configs directory.

- **Run benchmarks with optimized configs**

  Use the config directory produced above with `run_config_dir.py`:

```bash
python3 benchmark/raytune/run_config_dir.py \
  --configs-dir /path/to/optimal/configs \
  --max-new-tokens 100 \
  --max-context-length 32678 \
  --max-requests 2 \
  --actors-per-gpu 1 \
  --benchmark-results-dir ./bench_results/
```

### 2. Implementation of optimization

- **Config builders**: For each sparse attention method, a config builder constructs a `ResearchAttentionConfig` (masker stack, defaults, and metadata) for a given model/task/objective.
- **Search spaces**: Builders attach Ray Tune search spaces (e.g. `config.masker_configs[i].search_space`) to selected hyperparameters; `run_optimize_configs.py` passes these to Ray.
- **Validity checker**: Each builder defines a small validity checker that rejects invalid hyperparameter combinations early so trials can be skipped before running the benchmark.

High-level flow:

```text
(model, task, objectives, builder name)
                │
                ▼
         Config builder
      ┌─────────┴────────────────────────────┐
      │                                      │
      ▼                                      ▼
ResearchAttentionConfig          Ray Tune search_space attached
      │
      ▼
Ray Tune iterates over configs ──► validity checker ──►
      │                           │
      ├─ valid  ──► run benchmark trial
      └─ invalid ──► skip early (no trial)
```

### 3. Adding a new builder

- **Create a builder**: Copy an existing builder from `benchmark/raytune/config_builders/`, rename it, and adapt:
  - masker composition and default parameters
  - Ray Tune search spaces on the relevant hyperparameters
  - the validity checker logic for early exit on bad configs
- **Wire it up**:
  - Register the new builder name wherever builders are dispatched (e.g. builder registry/factory).
  - Add the new name to `BUILDER_NAMES` in `OPTIMIZATION_EXPERIMENT.py` so it is included in optimization and benchmarking.

**Example sketch (`vattention_pqcache`)** in `config_builders/vattention_pqcache.py` (Check the file for details) :

- **1. Builder name**:

  - Decorator: `@register_builder("vattention_pqcache")`
  - Class: `VAttentionPQCacheConfigBuilder`

- **2. Search space**:

  - Base definition on the PQCache masker:

    ```python
    config.masker_configs[2].search_space = {
        "pq_group_factor": tune.grid_search([2, 4]),
        "pq_bits": tune.grid_search([4, 8]),
        "kmeans_iter": tune.grid_search([10]),
        "metric": tune.grid_search(["euclidean"]),
    }
    ```

  - Plus sparsity-dependent grids on PQCache + AdaptiveSampling (e.g. `config.masker_configs[2].search_space["heavy_size"] = ...`, `config.masker_configs[3].search_space = {...}` inside the `if sparsity_objective == ...` blocks).

- **3. Validity checker**:

  - Function: `_validity_check(config, sparsity_val)` at the top of the file.
  - Attached to the config with:

    ```python
    config.validity_constraint = partial(_validity_check, sparsity_val=sparsity_val)
    ```


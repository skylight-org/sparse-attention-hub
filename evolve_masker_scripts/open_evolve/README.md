### Running openevolve based exploration

1. Use openvolve from here: (checkout into sparse-attention-hub/evolve_masker_scripts/open_evolve/openevolve)
```
https://github.com/algorithmicsuperintelligence/openevolve.git
```

In openevolve-run.py Make the following changes to handle GPU based evaluation

```
 """
 Entry point script for OpenEvolve
 """
+import torch
+import torch.multiprocessing as mp
+
 import sys
 from openevolve.cli import main

 if __name__ == "__main__":
+    mp.set_start_method('spawn')
     sys.exit(main())
```

2. Make necessary changes to  the evaluator.py 
     - what benchmark/dataset you want to use for validation
     - how many samples/token generations
     - what is the base config on which you want to build a new logic. (see sparse_attention_config.) (read more about maskers [here](https://github.com/skylight-org/sparse-attention-hub?tab=readme-ov-file#-what-are-masks-and-maskers))

3. Make necessary changes to config
We have a sample config as a starter config_1.yaml

4. Run open evolve and use open evolve utility (See openevolve repo) for visualization etc.

```
python evolve_masker_scripts/open_evolve/openevolve/openevolve-run.py  ./sparse_attention_hub/sparse_attention/research_attention/maskers/evolve/evolve_masker.py evolve_masker_scripts/evaluator.py --config evolve_masker_scripts/open_evolve/config_1.yaml --iterations <ITR>
```

### Dependencies:
```
cd sparse-attention-hub
pip install -e . && pip install -e .[dev]
```


### starting point.

This code would evolve over 
```
sparse-attention-hub/sparse_attention_hub/sparse_attention/research_attention/maskers/evolve/evolve_masker.py
```
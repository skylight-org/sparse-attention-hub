# Explore Various sparse attentions in chat mode.

### Install sparse attention hub

```
git clone https://github.com/skylight-org/sparse-attention-hub.git
cd sparse-attention-hub
pip install -e . && pip install -e .[dev]
pip flash-attn 
```

###  Use the config that you want to experiment with. Example config of vAttention(PQCache). Add it to the top of demo.py and ensure appropriate imports for maskers

```
    sparse_attention_config = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(
                sink_size=128,
            ),
            LocalMaskerConfig(
                window_size=128,
            ),
            PQCacheConfig(
                heavy_size=0.1,
                pq_group_factor=2,
                pq_bits=6,
                kmeans_iter=10,
                init_offset=128,
                metric="euclidean",
            ),
            AdaptiveSamplingMaskerConfig(
              delta = 0.05,
              epsilon = 0.05,
              init_offset = 128,
              local_offset =  128,
              base_rate_sampling = 0.05
            )

        ]
    )
```

### Run the model of your choice with sparse-attention-hub
```
python3 demo/chat.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 
```

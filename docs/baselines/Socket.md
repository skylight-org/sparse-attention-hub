# SOCKET: Soft Collision Kernel Estimator for Sparse Attention

## 1. Overview

**SOCKET (Soft Collision Kernel Estimator)** is a deterministic sparse attention
mechanism inspired by:

- RACE (Repeated Arrays of Count Estimators)
- Locality-Sensitive Hashing (LSH)
- Sign Random Projection (SRP)

Instead of computing full query–key dot products, SOCKET identifies a small set of
high-utility keys per query using **hash-based soft collisions**, and applies
attention only to those keys.

Key properties:

- Deterministic (no sampling)
- Query-dependent sparsity
- Value-aware ranking
- Efficient approximation to dense attention

---

## 2. Notation

| Symbol | Description |
|------|-------------|
| \( q \) | Query vector |
| \( k_i \) | Key vector at position \( i \) |
| \( v_i \) | Value vector at position \( i \) |
| \( L \) | Number of hash tables |
| \( P \) | Number of hyperplanes per table |
| \( R = 2^P \) | Number of buckets per table |
| \( M \) | Final sparse attention budget |

---

## 3. Hard Hashing of Keys (Sign Random Projection)

Each key is deterministically assigned to **one bucket per hash table** using
random hyperplanes.

### 3.1 Random hyperplanes

For each table ℓ ∈ {1, …, L}, sample P random vectors:

w_{ℓ,p} ~ N(0, I)


---

### 3.2 Projection and sign bits

For key \( k_i \):

z_{ℓ,p}(i) = ⟨k_i, w_{ℓ,p}⟩

b_{ℓ,p}(i) = 1[z_{ℓ,p}(i) ≥ 0]


---

### 3.3 Bucket index (big-endian encoding)

The bucket ID for key \( i \) in table \( \ell \) is:

bucket_ℓ(i) = ∑_{p=1}^{P} b_{ℓ,p}(i) · 2^{P − p}

Each key belongs to **exactly one bucket per table**.

---

## 4. Soft Hashing of Queries

Queries are *soft-assigned* to all buckets using the same hyperplanes.

### 4.1 Query projection

For query \( q \):

z_{ℓ,p}(q) = ⟨q, w_{ℓ,p}⟩

---

### 4.2 Nonlinear stabilization

h_{ℓ,p}(q) = tanh(z_{ℓ,p}(q))

---

### 4.3 Hypercube corners

All buckets correspond to vertices of a \( P \)-dimensional hypercube:

c_r ∈ {−1, +1}^P,   r = 1, …, R

---

### 4.4 Bucket logits

logits_{q,ℓ,r} = ∑_{p=1}^{P} h_{ℓ,p}(q) · c_r[p]


### 4.5 Softmax with temperature

P(r | q, ℓ) = softmax_r(logits_{q,ℓ,r} / τ)

This yields a probability distribution over buckets for each query and table.

---

## 5. Soft Collision Scoring

SOCKET does **not** form explicit candidate sets or Top-t buckets.
Instead, it computes **soft collision scores** directly.

### 5.1 Soft collision count

For query \( q \) and key \( i \):

C(q, i) = ∑_{ℓ=1}^{L} P(bucket_ℓ(i) | q, ℓ)


Interpretation:

- Each hash table contributes a soft vote
- Larger values indicate stronger query–key locality agreement

---

## 6. Value-Aware Ranking

To favor keys that can meaningfully influence the attention output, SOCKET
incorporates value magnitude.

### 6.1 Value norm

‖v_i‖₂ = √(∑_d v_{i,d}²)

---

### 6.2 Final score

score(q, i) = C(q, i) · ‖v_i‖₂


This balances:

- Structural similarity (hash collisions)
- Content strength (value magnitude)

---

## 7. Sparse Top-M Selection

For each query:

1. Compute \( \text{score}(q,i) \) for all keys
2. Mask out disallowed positions (padding / causal masks)
3. Select the **top-M** keys
4. Construct a binary sparse attention mask

The resulting mask defines a **deterministic, query-dependent sparse attention
pattern**.

---

## 8. Summary

SOCKET replaces quadratic attention with a structured sparse mechanism:

- Keys are hard-hashed via SRP
- Queries softly vote over hash buckets
- Soft collision scores approximate attention relevance
- Value magnitude refines ranking
- Top-M selection yields efficient sparse attention

This approach provides a scalable and effective approximation to dense
attention for long-context models.



### Example config in sparse-attention-hub
```
  config = ResearchAttentionConfig(masker_configs=[
      SinkMaskerConfig(sink_size=128),
      LocalMaskerConfig(window_size=128),
      SocketMaskerConfig(K=12, L=60, tau=0.3, heavy_size=0.2),
  ])
```

### Experimental Setup
Some datasets from the RULER benchmark.

Our Results with model - meta-llama/Llama-3.1-8B-Instruct:

| Dataset        | Token Budget 1600 (0.05) | Token Budget 3200 (0.10) | Token Budget 6400 (0.20) |
|:-------------- |:------------------------:|:-------------------------:|:-------------------------:|
| **vt**         |        91.4        |         94.2        |         95.2        |      
| **fwe**        |        86        |        88.7         |         91.7      |    
| **multikey_2** |       93         |         95        |          97       |   
| **qa_2**       |       82         |          53       |          53       |   
| **qa_1**       |       52        |          82       |          84       |  
| **multikey_3** |       92         |        100         |        100         |  



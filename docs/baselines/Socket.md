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

\[
z_{\ell,p}(i) = \langle k_i, w_{\ell,p} \rangle
\]

\[
b_{\ell,p}(i) = \mathbf{1}[z_{\ell,p}(i) \ge 0]
\]

---

### 3.3 Bucket index (big-endian encoding)

The bucket ID for key \( i \) in table \( \ell \) is:

\[
\text{bucket}_\ell(i)
= \sum_{p=1}^{P} b_{\ell,p}(i)\, 2^{P - p}
\]

Each key belongs to **exactly one bucket per table**.

---

## 4. Soft Hashing of Queries

Queries are *soft-assigned* to all buckets using the same hyperplanes.

### 4.1 Query projection

For query \( q \):

\[
z_{\ell,p}(q) = \langle q, w_{\ell,p} \rangle
\]

---

### 4.2 Nonlinear stabilization

\[
h_{\ell,p}(q) = \tanh(z_{\ell,p}(q))
\]

---

### 4.3 Hypercube corners

All buckets correspond to vertices of a \( P \)-dimensional hypercube:

\[
c_r \in \{-1,+1\}^P, \quad r = 1,\dots,R
\]

---

### 4.4 Bucket logits

\[
\text{logits}_{q,\ell,r}
= \sum_{p=1}^{P} h_{\ell,p}(q)\, c_r[p]
\]

---

### 4.5 Softmax with temperature

\[
P(r \mid q, \ell)
= \text{softmax}_r\left(\frac{\text{logits}_{q,\ell,r}}{\tau}\right)
\]

This yields a probability distribution over buckets for each query and table.

---

## 5. Soft Collision Scoring

SOCKET does **not** form explicit candidate sets or Top-t buckets.
Instead, it computes **soft collision scores** directly.

### 5.1 Soft collision count

For query \( q \) and key \( i \):

\[
C(q,i)
= \sum_{\ell=1}^{L}
P\big(\text{bucket}_\ell(i) \mid q, \ell\big)
\]

Interpretation:

- Each hash table contributes a soft vote
- Larger values indicate stronger query–key locality agreement

---

## 6. Value-Aware Ranking

To favor keys that can meaningfully influence the attention output, SOCKET
incorporates value magnitude.

### 6.1 Value norm

\[
\|v_i\|_2
= \sqrt{\sum_d v_{i,d}^2}
\]

---

### 6.2 Final score

\[
\text{score}(q,i)
= C(q,i) \cdot \|v_i\|_2
\]

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
      BucketMaskerConfig(K=12, L=60, tau=0.3, heavy_size=0.2),
  ])
```

### Experimental Setup
Some datasets from the RULER benchmark

In general, as the sparsity drops, there is a need to increase L (hash tables). 
  - Full recovery for 20% sparsity can be done with 30-32 tables.
  - Full recovery for 10% sparsity can be done with 50-52 tables.
  - Full recovery for 5% sparsity can be done with 78-80 tables.

Our Results with model - meta-llama/Llama-3.1-8B-Instruct:

| Dataset        | Token Budget 1600 (0.05) | Token Budget 3200 (0.10) | Token Budget 6400 (0.20) |
|:-------------- |:------------------------:|:-------------------------:|:-------------------------:|
| **vt**         |                |                 |         92        |
| **fwe**        |                |                 |         93.3      |
| **multikey_2** |                |         94        |          96       |
| **qa_2**       |                |          56       |          58       |
| **qa_1**       |                |          80       |          80       |
| **multikey_3** |       94         |        100         |        100         |



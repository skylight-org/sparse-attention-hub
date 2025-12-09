# Bucket Attention Sparse Attention Baseline

## 1. Overview

**Bucket Attention** is a sparse attention mechanism inspired by **RACE (Repeated Arrays of Count Estimators)** and **LSH-based soft hashing**.

Instead of evaluating all query–key dot products, Bucket Attention:

1. **Hard-hashes keys** into buckets using Sign Random Projection (SRP).  
2. **Soft-hashes queries** to obtain probability distributions over the same buckets.  
3. **Selects the top-t buckets** per query for each hash table.  
4. Builds a **candidate set** by taking the union of all keys that fall into selected buckets.  
5. Performs **value-aware-collision ranking** to recover the true Top-K candidates for attention.

---

## 2. Hard-Hashing Keys (Sign Random Projection)

We use **L independent hash tables**, each containing **P random hyperplanes**.

### 2.1 Projection onto hyperplanes

For a key vector $\( k_i \)$: $z_{\ell,p}(i) = \langle k_i,\ w_{\ell,p} \rangle$

### 2.2 Sign bit

$$
b_{\ell,p}(i) = \mathbf{1}[z_{\ell,p}(i) \ge 0]
$$

### 2.3 Bucket index (big-endian)

$$
\text{bucket}_\ell(i)
= \sum_{p=1}^{P} b_{\ell,p}(i)\ 2^{P - p}
$$


Keys that hash to the same bucket ID are treated as belonging to the same locality cluster.

---

## 3. Soft-Hashing Queries

Queries are "soft-assigned" to buckets using the same hyperplanes:

1. Project queries: $z_{\ell,p}(q)$
2. Apply nonlinearity: $\tanh(z_{\ell,p}(q))$
4. Compute similarity to all **R hypercube corners** $\( c_r \in \{-1,+1\}^P \)$:

$$
\text{logits}_{q,\ell,r}
= \sum_{p=1}^{P} \tanh(z_{\ell,p}(q)) \cdot c_r[p]
$$

A softmax yields per-table bucket probabilities:

$$
P(r \mid q, \ell) = \text{softmax}_r(\text{logits}_{q,\ell,r})
$$

## 5. Bucket Selection (Union of Matching Buckets Across Tables)

Once keys and queries have been hashed, Bucket Attention determines which keys
are *candidates* for each query by checking whether they land in any of the
query’s top-t buckets across the L hash tables.

### 5.1 Key–Query Bucket Matching

For each hash table ℓ:

- Each key `i` has a hard bucket assignment  
  $$
  \text{bucket}_\ell(i) \in \{0,\dots,R-1\}.
  $$

- Each query `q` has a list of **top-t buckets**:  
  $$
  \text{Top}_t(q,\ell) = \{r_1, \dots, r_t\}.
  $$

A key `i` is considered a match for query `q` in table ℓ if:

$$
\text{bucket}_\ell(i) \in \text{Top}_t(q,\ell).
$$

### 5.2 Candidate Mask

A key becomes a **candidate** if it matches in *any* of the L tables:

$$
\text{candidate}(q,i)
= \bigvee_{\ell = 1}^{L}\ \mathbf{1}\big[
\text{bucket}_\ell(i) \in \text{Top}_t(q,\ell)
\big].
$$


This mask represents the **union of all selected buckets** across tables.

### 5.3 Collision Counts

Beyond binary membership, we count how many tables vote for each key:

$$
\text{collisions}(q,i)
= \sum_{\ell=1}^{L}
\mathbf{1}\big[
\text{bucket}_\ell(i) \in \text{Top}_t(q,\ell)
\big].
$$

- If `collisions = 0`: the key was never selected.  
- If `collisions >= 1`: the key is a candidate.  
- If `collisions` is large: multiple tables agree that the key is relevant.

Collision counts behave as a **soft similarity signal**, often correlating with true attention weight.

---

## 6. Value-Aware Scoring (Final Ranking)

Candidate keys are then ranked before selecting the final top-K heavy tokens.
The ranking combines:

1. **Collision strength**  
2. **Value vector magnitude**

### 6.1 Value Norm

For each key value vector $\( v_i \)$:

$$
\|v_i\|_2
= \sqrt{\sum_{d} v_{i,d}^2}.
$$

This norm measures how much contribution the value vector can make to the
output—keys with larger values have greater influence.


### 6.2 Combined Collision Count + Value Score

The final score for query $\( q \)$ and key $\( i \)$ is:

$$
\text{score}(q,i)
= \text{collisions}(q,i)\ \cdot\ \|v_i\|_2.
$$

Interpretation:

- **High collision count ⇒ key is repeatedly hashed near the query**  
- **High value norm ⇒ key has potential to contribute strongly**  
- The product balances structural similarity (hashing) and content magnitude (values)

### 6.3 Top-K Selection

Among all candidate keys:

1. Mask out all non-candidates.
2. Select the **Top-K** keys by `score(q,i)` for each query to compute attention on.



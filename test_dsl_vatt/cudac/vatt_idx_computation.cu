#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

constexpr int64_t kHashP = 1000007;
constexpr int64_t kHashA = 2323;
constexpr int64_t kHashB = 2277;
constexpr int64_t kHashC = 1777;

__device__ __forceinline__ int64_t hash_index(
    int64_t i,
    int64_t start_idx,
    int64_t end_idx,
    int64_t b,
    int64_t q_head) {
  int64_t sampling_range = end_idx - start_idx;
  int64_t hash_val = (kHashA * i + kHashB * b + kHashC * q_head) % kHashP;
  int64_t idx = (hash_val % sampling_range) + start_idx;
  return idx;
}

template <typename scalar_t>
__device__ __forceinline__ float quantize(float value) {
  return static_cast<float>(static_cast<scalar_t>(value));
}

template <typename scalar_t>
__global__ void vatt_idx_kernel(
    const scalar_t* keys,
    const scalar_t* queries,
    int64_t* sparse_lens,
    int64_t* sparse_idx,
    scalar_t* weights,
    int64_t B,
    int64_t qH,
    int64_t kH,
    int64_t keys_len,
    int64_t D,
    int64_t base_sample_size,
    int64_t max_sample_size,
    float epsilon,
    float delta_ppf,
    float scaling,
    int64_t start_offset,
    int64_t end_offset,
    int64_t keys_stride_b,
    int64_t keys_stride_h,
    int64_t keys_stride_l,
    int64_t keys_stride_d,
    int64_t queries_stride_b,
    int64_t queries_stride_h,
    int64_t queries_stride_l,
    int64_t queries_stride_d,
    int64_t sparse_lens_stride_b,
    int64_t sparse_lens_stride_h,
    int64_t sparse_lens_stride_l,
    int64_t sparse_lens_stride_d,
    int64_t sparse_idx_stride_b,
    int64_t sparse_idx_stride_h,
    int64_t sparse_idx_stride_l,
    int64_t sparse_idx_stride_d,
    int64_t weights_stride_b,
    int64_t weights_stride_h,
    int64_t weights_stride_l,
    int64_t weights_stride_d) {
  int64_t q_head = static_cast<int64_t>(blockIdx.x);
  int64_t b = static_cast<int64_t>(blockIdx.y);
  int64_t z = static_cast<int64_t>(blockIdx.z);
  if (b >= B || q_head >= qH) {
    return;
  }

  int64_t k_head = q_head % kH;
  int64_t tid = static_cast<int64_t>(threadIdx.x);

  int64_t scores_count = start_offset + end_offset;
  int64_t start_idx = start_offset;
  int64_t end_idx = keys_len - end_offset;
  int64_t sampling_range = end_idx - start_idx;
  int64_t effective_max_sample =
      max_sample_size < sampling_range ? max_sample_size : sampling_range;

  extern __shared__ float shared_storage[];
  float* query_shared = shared_storage;
  float* scores_shared = query_shared + D;
  float* reduce_shared = scores_shared + scores_count;
  float* reduce2_shared = reduce_shared + blockDim.x;
  __shared__ int64_t sparse_len_shared;

  int64_t query_base = b * queries_stride_b + q_head * queries_stride_h;
  for (int64_t d = tid; d < D; d += blockDim.x) {
    int64_t q_offset = query_base + d * queries_stride_d;
    query_shared[d] = static_cast<float>(queries[q_offset]);
  }
  __syncthreads();

  for (int64_t idx = tid; idx < scores_count; idx += blockDim.x) {
    int64_t key_idx = idx < start_offset
        ? idx
        : (keys_len - end_offset + (idx - start_offset));
    int64_t key_base =
        b * keys_stride_b + k_head * keys_stride_h + key_idx * keys_stride_l;
    float dot = 0.0f;
    bool can_vec = (keys_stride_d == 1) && (D % 4 == 0) &&
        ((reinterpret_cast<uintptr_t>(query_shared) & 0xF) == 0) &&
        ((reinterpret_cast<uintptr_t>(keys + key_base) & 0xF) == 0);
    if (can_vec) {
      int64_t D4 = D / 4;
      const float4* q4 = reinterpret_cast<const float4*>(query_shared);
      const float4* k4 = reinterpret_cast<const float4*>(keys + key_base);
      for (int64_t d4 = 0; d4 < D4; ++d4) {
        float4 qv = q4[d4];
        float4 kv = k4[d4];
        dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
      }
    } else {
      for (int64_t d = 0; d < D; ++d) {
        float qv = query_shared[d];
        float kv = static_cast<float>(keys[key_base + d * keys_stride_d]);
        dot += qv * kv;
      }
    }
    scores_shared[idx] = quantize<scalar_t>(dot * scaling);
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int64_t idx = tid; idx < scores_count; idx += blockDim.x) {
    float score = scores_shared[idx];
    if (score > local_max) {
      local_max = score;
    }
  }
  reduce_shared[tid] = local_max;
  __syncthreads();
  for (int64_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float other = reduce_shared[tid + stride];
      if (other > reduce_shared[tid]) {
        reduce_shared[tid] = other;
      }
    }
    __syncthreads();
  }
  float max_norm = quantize<scalar_t>(reduce_shared[0]);
  __syncthreads();

  float local_sum = 0.0f;
  for (int64_t idx = tid; idx < scores_count; idx += blockDim.x) {
    float term = expf(scores_shared[idx] - max_norm);
    term = quantize<scalar_t>(term);
    local_sum += term;
  }
  reduce_shared[tid] = local_sum;
  __syncthreads();
  for (int64_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_shared[tid] += reduce_shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    reduce_shared[0] = quantize<scalar_t>(reduce_shared[0]);
  }
  __syncthreads();
  float static_denominator = reduce_shared[0];
  __syncthreads();

  float sum_x = 0.0f;
  float sum_x2 = 0.0f;
  for (int64_t i = tid; i < base_sample_size; i += blockDim.x) {
    int64_t idx = hash_index(i, start_idx, end_idx, b, q_head);
    int64_t key_base =
        b * keys_stride_b + k_head * keys_stride_h + idx * keys_stride_l;
    float dot = 0.0f;
    bool can_vec = (keys_stride_d == 1) && (D % 4 == 0) &&
        ((reinterpret_cast<uintptr_t>(query_shared) & 0xF) == 0) &&
        ((reinterpret_cast<uintptr_t>(keys + key_base) & 0xF) == 0);
    if (can_vec) {
      int64_t D4 = D / 4;
      const float4* q4 = reinterpret_cast<const float4*>(query_shared);
      const float4* k4 = reinterpret_cast<const float4*>(keys + key_base);
      for (int64_t d4 = 0; d4 < D4; ++d4) {
        float4 qv = q4[d4];
        float4 kv = k4[d4];
        dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
      }
    } else {
      for (int64_t d = 0; d < D; ++d) {
        float qv = query_shared[d];
        float kv = static_cast<float>(keys[key_base + d * keys_stride_d]);
        dot += qv * kv;
      }
    }
    float score = quantize<scalar_t>(dot * scaling);
    float attn = expf(score - max_norm);
    attn = quantize<scalar_t>(attn);
    sum_x += attn;
    float attn_sq = quantize<scalar_t>(attn * attn);
    sum_x2 += attn_sq;
  }
  reduce_shared[tid] = sum_x;
  reduce2_shared[tid] = sum_x2;
  __syncthreads();
  for (int64_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_shared[tid] += reduce_shared[tid + stride];
      reduce2_shared[tid] += reduce2_shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    reduce_shared[0] = quantize<scalar_t>(reduce_shared[0]);
    reduce2_shared[0] = quantize<scalar_t>(reduce2_shared[0]);
  }
  __syncthreads();

  if (tid == 0) {
    float inv_base = 1.0f / static_cast<float>(base_sample_size);
    float ex = quantize<scalar_t>(reduce_shared[0] * inv_base);
    float ex2 = quantize<scalar_t>(reduce2_shared[0] * inv_base);
    float ex_sq = quantize<scalar_t>(ex * ex);
    float var = quantize<scalar_t>(ex2 - ex_sq);
    float min_var = quantize<scalar_t>(1e-8f);
    if (var < min_var) {
      var = min_var;
    }
    float sampling_range_f = static_cast<float>(sampling_range);
    float estimated_denominator = quantize<scalar_t>(ex * sampling_range_f);
    float total_denominator =
        quantize<scalar_t>(static_denominator + estimated_denominator);
    float epsilon_allowable = quantize<scalar_t>(epsilon * total_denominator);
    float min_eps = quantize<scalar_t>(1e-8f);
    if (epsilon_allowable < min_eps) {
      epsilon_allowable = min_eps;
    }
    float ratio = quantize<scalar_t>(
        (delta_ppf * sampling_range_f) / epsilon_allowable);
    float ratio_sq = quantize<scalar_t>(ratio * ratio);
    float budget = quantize<scalar_t>(ratio_sq * var);
    float min_budget = quantize<scalar_t>(
        static_cast<float>(base_sample_size));
    float max_budget = quantize<scalar_t>(
        static_cast<float>(effective_max_sample));
    if (budget < min_budget) {
      budget = min_budget;
    }
    if (budget > max_budget) {
      budget = max_budget;
    }
    sparse_len_shared = static_cast<int64_t>(budget);
    if (z == 0) {
      int64_t out_offset =
          b * sparse_lens_stride_b + q_head * sparse_lens_stride_h +
          0 * sparse_lens_stride_l + 0 * sparse_lens_stride_d;
      sparse_lens[out_offset] = sparse_len_shared;
    }
  }
  __syncthreads();

  int64_t sparse_len = sparse_len_shared;
  if (sparse_len <= 0) {
    return;
  }
  int64_t grid_z = static_cast<int64_t>(gridDim.z);
  int64_t chunk = (sparse_len + grid_z - 1) / grid_z;
  int64_t start = z * chunk;
  int64_t end = start + chunk;
  if (end > sparse_len) {
    end = sparse_len;
  }
  if (start >= sparse_len) {
    return;
  }

  float weight_value = static_cast<float>(sampling_range) /
      static_cast<float>(sparse_len);
  for (int64_t i = start + tid; i < end; i += blockDim.x) {
    int64_t idx = hash_index(i, start_idx, end_idx, b, q_head);
    int64_t out_offset =
        b * sparse_idx_stride_b + q_head * sparse_idx_stride_h +
        0 * sparse_idx_stride_l + i * sparse_idx_stride_d;
    sparse_idx[out_offset] = idx;
    int64_t weight_offset =
        b * weights_stride_b + q_head * weights_stride_h +
        0 * weights_stride_l + i * weights_stride_d;
    weights[weight_offset] = static_cast<scalar_t>(weight_value);
  }
}

}  // namespace

std::vector<torch::Tensor> vatt_idx_computation_cuda(
    torch::Tensor keys,
    torch::Tensor queries,
    int64_t base_sample_size,
    int64_t max_sample_size,
    double epsilon,
    double delta_ppf,
    double scaling,
    int64_t start_offset,
    int64_t end_offset) {
  TORCH_CHECK(keys.is_cuda(), "keys must be a CUDA tensor");
  TORCH_CHECK(queries.is_cuda(), "queries must be a CUDA tensor");
  TORCH_CHECK(keys.scalar_type() == queries.scalar_type(),
              "keys and queries must have the same dtype");
  TORCH_CHECK(keys.scalar_type() == torch::kFloat,
              "keys and queries must be float32");
  TORCH_CHECK(queries.size(2) == 1, "queries length dimension must be 1");
  TORCH_CHECK(start_offset + end_offset > 0,
              "start_offset + end_offset must be > 0");

  int64_t B = keys.size(0);
  int64_t kH = keys.size(1);
  int64_t keys_len = keys.size(2);
  int64_t D = keys.size(3);
  int64_t qH = queries.size(1);
  TORCH_CHECK(queries.size(0) == B, "queries batch size must match keys");
  TORCH_CHECK(end_offset <= keys_len, "end_offset must be <= keys length");
  TORCH_CHECK(start_offset < keys_len, "start_offset must be < keys length");

  int64_t start_idx = start_offset;
  int64_t end_idx = keys_len - end_offset;
  TORCH_CHECK(end_idx > start_idx, "sampling range must be positive");
  TORCH_CHECK(base_sample_size > 0, "base_sample_size must be > 0");

  torch::Tensor keys_contig = keys.contiguous();
  torch::Tensor queries_contig = queries.contiguous();

  torch::TensorOptions idx_options =
      torch::TensorOptions().dtype(torch::kInt64).device(keys.device());
  torch::TensorOptions weight_options =
      torch::TensorOptions().dtype(keys.scalar_type()).device(keys.device());

  torch::Tensor sparse_lens = torch::zeros({B, qH, 1, 1}, idx_options);
  torch::Tensor sparse_idx = torch::zeros({B, qH, 1, keys_len}, idx_options);
  torch::Tensor weights = torch::zeros({B, qH, 1, keys_len}, weight_options);

  constexpr int64_t block_size = 256;
  dim3 grid(static_cast<unsigned int>(qH),
            static_cast<unsigned int>(B),
            1);
  dim3 block(static_cast<unsigned int>(block_size));
  int64_t scores_count = start_offset + end_offset;
  size_t shared_bytes =
      static_cast<size_t>(D + scores_count + 2 * block_size) * sizeof(float);

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  vatt_idx_kernel<float><<<grid, block, shared_bytes, stream>>>(
      keys_contig.data_ptr<float>(),
      queries_contig.data_ptr<float>(),
      sparse_lens.data_ptr<int64_t>(),
      sparse_idx.data_ptr<int64_t>(),
      weights.data_ptr<float>(),
      B,
      qH,
      kH,
      keys_len,
      D,
      base_sample_size,
      max_sample_size,
      static_cast<float>(epsilon),
      static_cast<float>(delta_ppf),
      static_cast<float>(scaling),
      start_offset,
      end_offset,
      keys_contig.stride(0),
      keys_contig.stride(1),
      keys_contig.stride(2),
      keys_contig.stride(3),
      queries_contig.stride(0),
      queries_contig.stride(1),
      queries_contig.stride(2),
      queries_contig.stride(3),
      sparse_lens.stride(0),
      sparse_lens.stride(1),
      sparse_lens.stride(2),
      sparse_lens.stride(3),
      sparse_idx.stride(0),
      sparse_idx.stride(1),
      sparse_idx.stride(2),
      sparse_idx.stride(3),
      weights.stride(0),
      weights.stride(1),
      weights.stride(2),
      weights.stride(3));

  return {sparse_lens, sparse_idx, weights};
}

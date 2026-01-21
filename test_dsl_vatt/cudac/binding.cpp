#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> vatt_idx_computation_cuda(
    torch::Tensor keys,
    torch::Tensor queries,
    int64_t base_sample_size,
    int64_t max_sample_size,
    double epsilon,
    double delta_ppf,
    double scaling,
    int64_t start_offset,
    int64_t end_offset);

std::vector<torch::Tensor> ref_vatt_idx_computation(
    torch::Tensor keys,
    torch::Tensor queries,
    int64_t base_sample_size,
    int64_t max_sample_size,
    double epsilon,
    double delta_ppf,
    double scaling,
    int64_t start_offset,
    int64_t end_offset) {
  return vatt_idx_computation_cuda(
      keys,
      queries,
      base_sample_size,
      max_sample_size,
      epsilon,
      delta_ppf,
      scaling,
      start_offset,
      end_offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "ref_vatt_idx_computation",
      &ref_vatt_idx_computation,
      "VATT idx computation (CUDA)");
}

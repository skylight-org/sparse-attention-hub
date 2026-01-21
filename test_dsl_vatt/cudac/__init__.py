"""CUDA extension bindings for vatt idx computation."""

from test_dsl_vatt.cudac.vatt_idx_computation import ref_vatt_idx_computation

__all__: list[str] = ["ref_vatt_idx_computation"]

#!/usr/bin/env python3
# question_and_solution_check.py

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# Question (reference model)
# ----------------------------
class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [
        torch.rand(batch_size, *input_shape),
        torch.randint(0, 2, (batch_size,)).float() * 2 - 1,
    ]

def get_init_inputs():
    return []

# ----------------------------
# Solution (with kernel code)
# ----------------------------
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add_cpp_source = "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"

elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.elementwise_add = elementwise_add

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

# ----------------------------
# Fast correctness check
# ----------------------------
def fast_check():
    ref = Model()
    cand = ModelNew()

    inputs = get_inputs()
    with torch.inference_mode():
        y_ref = ref(*inputs)
        y_cand = cand(*inputs)

    if torch.allclose(y_ref, y_cand, rtol=1e-6, atol=1e-6):
        print("PASS ✅")
    else:
        print("FAIL ❌")
        print("ref:", y_ref.item())
        print("cand:", y_cand.item())
        print("diff:", (y_ref - y_cand).abs().max().item())

if __name__ == "__main__":
    fast_check()

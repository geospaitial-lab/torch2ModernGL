#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor gl_texture_to_torch_cu(
    const int gl_object,
    const int width,
    const int height,
    const int components,
    const int dtype
);

void torch_to_gl_texture_cu(
    const torch::Tensor tensor,
    const int gl_object,
    const int width,
    const int height,
    const int element_bytes
);
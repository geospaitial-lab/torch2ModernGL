#include "utils.h"

torch::Tensor gl_texture_to_torch(
    const int gl_object,
    const int width,
    const int height,
    const int components,
    const int dtype
){
    return gl_texture_to_torch_cu(gl_object, width, height, components, dtype);
}

void torch_to_gl_texture(
    const torch::Tensor tensor,
    const int gl_object,
    const int width,
    const int height,
    const int element_bytes
){
    CHECK_INPUT(tensor);
    torch_to_gl_texture_cu(tensor, gl_object, width, height, element_bytes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("gl_texture_to_torch", &gl_texture_to_torch);
    m.def("torch_to_gl_texture", &torch_to_gl_texture);
}
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#endif
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void deleter(void* device_pointer){
    gpuErrchk(cudaFree(device_pointer));
}

torch::Tensor gl_texture_to_torch_cu(
    const int gl_object,
    const int width,
    const int height,
    const int components,
    const int dtype
){

    int bytes = 1;
    auto options = torch::TensorOptions().device(torch::kCUDA);
    if(dtype==0){
        bytes = 1;
        options = options.dtype(torch::kUInt8);
    } else if(dtype==1){
        bytes = 4;
        options = options.dtype(torch::kFloat32);
    } else if(dtype==2){
        bytes = 1;
        options = options.dtype(torch::kInt8);
    } else if(dtype==3){
        bytes = 2;
        options = options.dtype(torch::kInt16);
    } else if(dtype==4){
        bytes = 4;
        options = options.dtype(torch::kInt32);
    } else {
    printf("Invalid dtype!");
    }

    struct cudaGraphicsResource *cuda_texture;
    gpuErrchk(cudaGraphicsGLRegisterImage(&cuda_texture, gl_object, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    gpuErrchk(cudaGraphicsMapResources(1, &cuda_texture, 0));

    cudaArray_t cuda_texture_device_array;
    gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&cuda_texture_device_array, cuda_texture, 0, 0));

	void* devicePtr = nullptr;
	gpuErrchk(cudaMalloc(&devicePtr, width * height * components * bytes));

	gpuErrchk(cudaMemcpy2DFromArray(devicePtr, width * components * bytes, cuda_texture_device_array
	    , 0, 0, width * components * bytes, height, cudaMemcpyDeviceToDevice));

	auto out_tensor = torch::from_blob(devicePtr, {height, width, components},
	    deleter, options);

	devicePtr = nullptr;

    gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_texture, 0));

    gpuErrchk(cudaGraphicsUnregisterResource(cuda_texture));

    return out_tensor;
}


void torch_to_gl_texture_cu(
    const torch::Tensor tensor,
    const int gl_object,
    const int width,
    const int height,
    const int element_bytes
){
    struct cudaGraphicsResource *cuda_texture;
    gpuErrchk(cudaGraphicsGLRegisterImage(&cuda_texture, gl_object, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard));
    gpuErrchk(cudaGraphicsMapResources(1, &cuda_texture, 0));

    cudaArray_t cuda_texture_device_array;
    gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&cuda_texture_device_array, cuda_texture, 0, 0));

	gpuErrchk(cudaMemcpy2DToArray(cuda_texture_device_array, 0, 0, tensor.data_ptr(), width * element_bytes,
	    width * element_bytes, height, cudaMemcpyDeviceToDevice));

    gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_texture, 0));

    gpuErrchk(cudaGraphicsUnregisterResource(cuda_texture));
}
// T9 scratch: fp32 -> fp16 result cast for the P40 daemon.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void p40_f32_to_f16_kernel(const float* __restrict__ in,
                                      __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

extern "C" void p40_f32_to_f16(const float* in, __half* out, int n, cudaStream_t s) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    p40_f32_to_f16_kernel<<<blocks, threads, 0, s>>>(in, out, n);
}

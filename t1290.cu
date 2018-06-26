#include <iostream>
#include <stdio.h>
#include <math.h>

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__inline__ __device__
float blockReduceSum(float val) {

    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__global__ void deviceReduceKernel(float *in, float *out, size_t N) {
    float sum = 0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}

__global__ void deviceReduceWarpAtomicKernel(float *in, float *out, int N) {
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {

        sum += in[i] + in[i + 1] + in[i + 2];
    }
    sum = warpReduceSum(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(out, sum);
}

__global__ void deviceReduceBlockAtomicKernel(float *in, float *out, float *out_j, int N, int rank) {
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        float sum_i = in[i] + in[i + 1] + in[i + 2];
        sum += sum_i;

        for (int j = 0; j < rank; j++) {
            float jacobi = -2.0 * sum_i;

            jacobi = blockReduceSum(jacobi);

            if (threadIdx.x == 0)
                atomicAdd(&out_j[j], jacobi);
        }
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}

int main() {
    int n = 5000;
    float b;
    float *a, *a_host;
    int rank = 10;
    int size = n * 3;
    float j[rank];
    a_host = new float[size];

    int blocks = (n / 512) + 1;

    float *d_b;
    cudaMalloc(&d_b, sizeof(float));

    float *d_j;
    cudaMalloc(&d_j, rank * sizeof(float));

    cudaMalloc((void **) &a, size * sizeof(float));
    for (int i = 0; i < n; i++) {
        a_host[i] = 1;
        a_host[i + 1] = 1;
        a_host[i + 2] = 1;
    }

    cudaMemcpy(a, a_host, size * sizeof(float), cudaMemcpyHostToDevice);

    deviceReduceBlockAtomicKernel << < blocks, 512 >> > (a, d_b, d_j, n, rank);
    //deviceReduceKernel<<<1, 1024>>>(d_intermediate, a, blocks);
    cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&j, d_j, rank*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_b);
    cudaFree(d_j);
    std::cout << "Result: " << b << std::endl;
    std::cout << "Jacobi:";
    for( int i = 1; i < rank; i++) {
        std::cout << " " << j[i];
    }
    std::cout << std::endl;

    return 0;
}

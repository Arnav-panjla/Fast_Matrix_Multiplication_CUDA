#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 16
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void matrixMulTiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        int tiledRow = row;
        int tiledColA = tile * BLOCK_SIZE + threadIdx.x;
        int tiledRowB = tile * BLOCK_SIZE + threadIdx.y;
        int tiledCol = col;

        Asub[threadIdx.y][threadIdx.x] = (tiledRow < N && tiledColA < N) ? A[tiledRow * N + tiledColA] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] = (tiledRowB < N && tiledCol < N) ? B[tiledRowB * N + tiledCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void init_matrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);

    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);
    float* h_c_cpu = (float*)malloc(size);
    float* h_c_gpu = (float*)malloc(size);


    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));

    srand((unsigned int)time(NULL));
    init_matrix(h_a, N);
    init_matrix(h_b, N);

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Matrix size: %dx%d\n", N, N);
    printf("Block size: %dx%d\n", BLOCK_SIZE, BLOCK_SIZE);
    // printf("Grid size: %dx%d\n", gridSize.x, gridSize.y);

    // Timing analysis
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMulTiled<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    cudaEventRecord(stop);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    printf("GPU kernel execution time: %.3f ms\n", milliseconds);
    printf("GPU kernel execution time: %.6f seconds\n", milliseconds / 1000.0f);

    float gflops = (2.0f * N * N * N) / (milliseconds / 1000.0f) / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Copying result back to host...\n");
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

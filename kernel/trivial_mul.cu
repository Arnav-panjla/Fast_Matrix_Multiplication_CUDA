# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include <time.h>


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void matrixMul(const float *a, float *b, float *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++){
            sum += a[row * N + i ] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

void init_matrix(float *mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    int N = 1024;

    size_t size = N * N * sizeof(float); // matrix size

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    init_matrix(h_a, N);
    init_matrix(h_b, N);    
    
    printf("Copying data to GPU...\n");
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice); 

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    printf("Matrix size: %dx%d\n", N, N);
    printf("Block size: %dx%d\n", blockSize.x, blockSize.y);
    printf("Grid size: %dx%d\n", gridSize.x, gridSize.y);

    // Timing analysis
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    matrixMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    cudaEventRecord(stop);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU kernel execution time: %.3f ms\n", milliseconds);
    printf("GPU kernel execution time: %.6f seconds\n", milliseconds / 1000.0f);

    float gflops = (2.0f * N * N * N) / (milliseconds / 1000.0f) / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Copying result back to host...\n");
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

void matrix_mul_cpu(const float *a, const float *b, float *c, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += a[row * N + i] * b[i * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

void init_matrix(float *mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);

    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    srand(time(NULL));
    init_matrix(h_a, N);
    init_matrix(h_b, N);

    printf("Matrix size: %dx%d\n", N, N);
    printf("Starting CPU matrix multiplication...\n");

    double start_time = get_time();
    
    matrix_mul_cpu(h_a, h_b, h_c, N);
    
    double end_time = get_time();
    double elapsed_time = end_time - start_time;

    printf("CPU execution time: %.3f ms\n", elapsed_time * 1000.0);
    printf("CPU execution time: %.6f seconds\n", elapsed_time);

    double gflops = (2.0 * N * N * N) / elapsed_time / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    printf("CPU matrix multiplication completed successfully!\n");
    return 0;
}

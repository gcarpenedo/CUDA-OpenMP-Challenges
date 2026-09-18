// DON'T CHANGE THIS ^^ FILENAME!
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "kernel.h"

// utility for wrapping CUDA API calls and log any error they may return (use this for debugging)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// === DO NOT CHANGE THIS ===
void initWith(float number, float* arr, int size) {
  for (int i = 0; i < size; i++) arr[i] = number;
}

void initRandom(float* arr, int size, unsigned int seed, float minVal = 0.0f, float maxVal = 1.0f) {
  srand(seed);
  for (int i = 0; i < size; i++) {
    float r = (float)rand() / RAND_MAX;
    arr[i] = minVal + r * (maxVal - minVal);
  }
}
// ==========================



int main(int argc, char** argv) {
  // === DO NOT CHANGE THIS ===
  if (argc != 6) {
    printf("Usage: %s <m> <k> <n> <batch> <seed>\n", argv[0]);
    exit(1);
  }

  int m = atoi(argv[1]); // rows of Ms and Ps
  int k = atoi(argv[2]); // cols of Ms, rows of Ns
  int n = atoi(argv[3]); // cols of Ns and Ps
  int batch = atoi(argv[4]); // number of matrix pairs
  unsigned int seed = (unsigned int)atoi(argv[5]); // seed for random initialization

  printf("Running batched matmul with m=%d, k=%d, n=%d, batch=%d, seed=%u\n", m, k, n, batch, seed);

  const int sizeM = m*k;
  const int sizeN = k*n*batch;
  const int sizeP = m*n*batch;

  float* M = (float*)malloc(sizeM * sizeof(float));
  float* N = (float*)malloc(sizeN * sizeof(float));
  float* P = (float*)malloc(sizeP * sizeof(float));

  initRandom(M, sizeM, seed);
  initRandom(N, sizeN, seed + 1);
  initWith(0.0f, P, sizeP);

  // ==========================

  float *M_d;
  float *N_d;
  float *P_d;

  gpuErrchk(cudaMalloc((void**)&M_d, sizeM * sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&N_d, sizeN * sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&P_d, sizeP * sizeof(float)));

  gpuErrchk(cudaMemcpy(M_d, M, sizeM * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(N_d, N, sizeN * sizeof(float), cudaMemcpyHostToDevice));

  dim3 blockSize(TILE_SIZE / MICRO_TILE_X, TILE_SIZE / MICRO_TILE_Y);
  dim3 numBlocks((n + TILE_SIZE - 1) / TILE_SIZE,  // Each block processes 1 tile
                 (m + TILE_SIZE - 1) / TILE_SIZE, 
                 batch);

  batchedMatMul<<<numBlocks, blockSize>>>(M_d, N_d, P_d, m, k, n, batch);

  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Ops and memory model (for reporting)
  long long fma_ops = (long long)m * n * k * batch;
  long long mem_read_bytes = ((long long)m*k + (long long)k*n*batch) * sizeof(float);
  long long mem_write_bytes = ((long long)m*n*batch) * sizeof(float);

  // Timing
  cudaEvent_t start, stop;
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk(cudaEventRecord(start));
  batchedMatMul<<<numBlocks, blockSize>>>(M_d, N_d, P_d, m, k, n, batch);
  gpuErrchk(cudaEventRecord(stop));
  gpuErrchk(cudaDeviceSynchronize());
  float ms = 0.0f;
  gpuErrchk(cudaEventElapsedTime(&ms, start, stop));

  // Copy back AFTER timing
  gpuErrchk(cudaMemcpy(P, P_d, sizeP * sizeof(float), cudaMemcpyDeviceToHost));

  // Simple benchmark output
  printf("Ops.FMA=%lld, Mem.ReadB=%lld, Mem.WriteB=%lld, Time.ms=%.3f\n",
         fma_ops, mem_read_bytes, mem_write_bytes, ms);

  // Save results to files for verification
  FILE* fM = fopen("M.bin", "wb");
  FILE* fN = fopen("N.bin", "wb");
  FILE* fP = fopen("P_gpu.bin", "wb");
  FILE* fParams = fopen("params.txt", "w");

  fwrite(M, sizeof(float), sizeM, fM);
  fwrite(N, sizeof(float), sizeN, fN);
  fwrite(P, sizeof(float), sizeP, fP);
  fprintf(fParams, "%d %d %d %d %u\n", m, k, n, batch, seed);

  fclose(fM);
  fclose(fN);
  fclose(fP);
  fclose(fParams);

  printf("Results saved to M.bin, N.bin, P_gpu.bin, params.txt\n");

  gpuErrchk(cudaEventDestroy(start));
  gpuErrchk(cudaEventDestroy(stop));
  gpuErrchk(cudaFree(M_d));
  gpuErrchk(cudaFree(N_d));
  gpuErrchk(cudaFree(P_d));
  free(M); free(N); free(P);
  return 0;
}

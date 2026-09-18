#ifndef KERNEL_H
#define KERNEL_H

// Tiling configuration
#define TILE_SIZE 64
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 4
#define MICRO_TILE_X 2  // iterations on X per thread
#define MICRO_TILE_Y 16 // iterations on Y per thread

__global__ void batchedMatMul(float* M, float* N, float* P, int m, int k, int n, int batch);

#endif // KERNEL_H

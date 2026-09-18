#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "kernel.h"

// Tiling configuration
#define TILE_SIZE 64
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 4
#define MICRO_TILE_X 2  // iterations on X per thread
#define MICRO_TILE_Y 16 // iterations on Y per thread

__global__ void batchedMatMul(float* M, float* N, float* P, int m, int k, int n, int batch) {
  __shared__ float ds_M[TILE_SIZE][TILE_SIZE];
  __shared__ float ds_N[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int b = blockIdx.z;

  int row_base = blockIdx.y * TILE_SIZE + ty * MICRO_TILE_Y;
  int col_base = blockIdx.x * TILE_SIZE + tx * MICRO_TILE_X;

  float values[MICRO_TILE_Y][MICRO_TILE_X]; // accumula i risultati parziali
  #pragma unroll
  for (int i = 0; i < MICRO_TILE_Y; i++)
    for (int j = 0; j < MICRO_TILE_X; j++)
      values[i][j] = 0.0f; // inizializzo

  int numTiles = (k + TILE_SIZE - 1) / TILE_SIZE; // numero di tile su k
  for (int p = 0; p < numTiles; ++p) { // ciclo sui tile di k
    #pragma unroll
    for (int i = 0; i < MICRO_TILE_Y; i++) { // carico M
      #pragma unroll
      for (int j = 0; j < MICRO_TILE_X; j++) {
        int sm_row = ty * MICRO_TILE_Y + i;
        int sm_col = tx * MICRO_TILE_X + j;
        int global_row = blockIdx.y * TILE_SIZE + sm_row;
        int global_col = p * TILE_SIZE + sm_col;
        if (global_row < m && global_col < k) {
          ds_M[sm_row][sm_col] = M[global_row * k + global_col];
        } else {
          ds_M[sm_row][sm_col] = 0.0f;
        }
      }
    }

    #pragma unroll
    for (int i = 0; i < MICRO_TILE_Y; i++) { // carico N
      #pragma unroll
      for (int j = 0; j < MICRO_TILE_X; j++) {
        int sm_row = ty * MICRO_TILE_Y + i;
        int sm_col = tx * MICRO_TILE_X + j;
        int global_row = p * TILE_SIZE + sm_row;
        int global_col = blockIdx.x * TILE_SIZE + sm_col;
        if (global_row < k && global_col < n && b < batch) {
          ds_N[sm_row][sm_col] = N[b * (k * n) + global_row * n + global_col];
        } else {
          ds_N[sm_row][sm_col] = 0.0f;
        }
      }
    }

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < TILE_SIZE; ++kk) { // kk scorre sulle colonne di M/righe di N nel tile
      #pragma unroll
      for (int i = 0; i < MICRO_TILE_Y; i++) { // ciclo sulle righe del micro-tile
        #pragma unroll
        for (int j = 0; j < MICRO_TILE_X; j++) { // ciclo sulle colonne del micro-tile
          float a = ds_M[ty * MICRO_TILE_Y + i][kk];
          float c = ds_N[kk][tx * MICRO_TILE_X + j];
          values[i][j] += a * c;
        }
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < MICRO_TILE_Y; i++) {
    #pragma unroll
    for (int j = 0; j < MICRO_TILE_X; j++) {
      int global_row = row_base + i;
      int global_col = col_base + j;
      if (global_row < m && global_col < n && b < batch) {
        P[b * (m * n) + global_row * n + global_col] = values[i][j];
      }
    }
  }
}
#include "utils.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <vector>

// HELPER FUNCTION [YOU CAN USE IT AS-IS]
// hue averaging using circular mean around the unit circle
// h: surrounding cells
// n: number of alive neighbors
float hue_average(const float *h, int n) {
  float sumx = 0.0f, sumy = 0.0f;
  for (int i = 0; i < n; i++) {
    float angle = h[i] * 2.0f * M_PI;
    sumx += cosf(angle);
    sumy += sinf(angle);
  }
  float angle = atan2f(sumy, sumx);
  if (angle < 0) angle += 2.0f * M_PI;
  return angle / (2.0f * M_PI);
}

// hue average implemented in a more efficient way
float hue_avg_better(const float *h, int n){
  const float two_pi = 2.0f * float(M_PI);
  float sumx = 0.0f;
  float sumy = 0.0f;

  for (int i = 0; i < n; ++i) {
    float angle = h[i] * two_pi;
    float s, c;
    // this operation shuould save some time, but is not necessary
    __builtin_sincosf(angle, &s, &c);
    sumx += c;
    sumy += s;
  }

  float angle = atan2f(sumy, sumx);
  if (angle < 0.0f) angle += two_pi;
  return angle / two_pi;
}

// HELPER FUNCTION [YOU DON'T NEED TO USE THIS]
// random hue selection
float pick_group_hue(int group_id) {
  float base = float(group_id) / NUM_GROUPS;
  float h = base + ((float)rand() / RAND_MAX)*2*COLOR_VARIATION - COLOR_VARIATION;
  if (h < 0) h += 1.0f;
  if (h >= 1) h -= 1.0f;
  return h;
}

// HELPER FUNCTION [YOU DON'T NEED TO USE THIS]
// grid initialization with a few clusters
void initialize_grid(Grid &g) {
  int W = g.W, H = g.H;

  int min_dim = (W < H ? W : H);
  int r = int(min_dim * CLUSTER_RADIUS_FACTOR);
  if (r < CLUSTER_MIN_RADIUS) r = CLUSTER_MIN_RADIUS;

  std::vector<float> group_hue(NUM_GROUPS);
  for (int i = 0; i < NUM_GROUPS; i++)
    group_hue[i] = pick_group_hue(i);

  std::vector<int> cx(NUM_GROUPS), cy(NUM_GROUPS);
  for (int i = 0; i < NUM_GROUPS; i++) {
    cx[i] = rand() % W;
    cy[i] = rand() % H;
  }

  for (int g_id = 0; g_id < NUM_GROUPS; g_id++) {
    float base_h = group_hue[g_id];
    int gx = cx[g_id], gy = cy[g_id];

    for (int dx = -r; dx <= r; dx++) {
      for (int dy = -r; dy <= r; dy++) {
        if (dy*dy + dx*dx <= r*r) {
          int x = (gx + dx + 2*W) % W, y = (gy + dy + 2*H) % H;
          if (((float)rand()/RAND_MAX) < CLUSTER_FILL_DENSITY) {
            int idx = y*W + x;
            g.alive[idx] = 1;
            g.hue[idx] = base_h;
          }
        }
      }
    }
  }
}

void initialize_grid(Grid_better &g) {
  int W = g.W, H = g.H;

  int min_dim = (W < H ? W : H);
  int r = int(min_dim * CLUSTER_RADIUS_FACTOR);
  if (r < CLUSTER_MIN_RADIUS) r = CLUSTER_MIN_RADIUS;

  std::vector<float> group_hue(NUM_GROUPS);
  for (int i = 0; i < NUM_GROUPS; i++)
    group_hue[i] = pick_group_hue(i);

  std::vector<int> cx(NUM_GROUPS), cy(NUM_GROUPS);
  for (int i = 0; i < NUM_GROUPS; i++) {
    cx[i] = rand() % W;
    cy[i] = rand() % H;
  }

  std::fill(g.alive.begin(), g.alive.end(), 0);
  std::fill(g.hue.begin(), g.hue.end(), 0.0f);
  std::fill(g.alive_neighbors.begin(), g.alive_neighbors.end(), 0);

  for (int g_id = 0; g_id < NUM_GROUPS; g_id++) {
    float base_h = group_hue[g_id];
    int gx = cx[g_id], gy = cy[g_id];

    for (int dx = -r; dx <= r; dx++) {
      for (int dy = -r; dy <= r; dy++) {
        if (dy*dy + dx*dx <= r*r) {
          int x = (gx + dx + 2*W) % W, y = (gy + dy + 2*H) % H;
          if (((float)rand()/RAND_MAX) < CLUSTER_FILL_DENSITY) {
            int idx = y*W + x;
            g.alive[idx] = 1;
            g.hue[idx] = base_h;
          }
        }
      }
    }
  }
}

// HELPER FUNCTION [YOU DON'T NEED TO USE THIS]
// compares two grids
bool compare_grids(const Grid &a, const Grid &b) {
  int N = a.W * a.H;
  for (int i = 0; i < N; i++) {
    if (a.alive[i] != b.alive[i]) return false;
    if (a.alive[i]) {
      float ha = a.hue[i];
      float hb = b.hue[i];
      // compare hue with a tolerance on floats (due to associativity)
      if (fabs(ha - hb) > 1e-4f) return false;
    }
  }
  return true;
}

bool compare_grids(const Grid &a, const Grid_better &b) {
  int N = a.W * a.H;
  for (int i = 0; i < N; i++) {
    if (a.alive[i] != b.alive[i]) return false;
    if (a.alive[i]) {
      float ha = a.hue[i];
      float hb = b.hue[i];
      if (fabs(ha - hb) > 1e-4f) return false;
    }
  }
  return true;
}

bool compare_grids(const Grid_better &a, const Grid &b) {
  return compare_grids(b, a);
}

// HELPER FUNCTION [YOU DON'T NEED TO USE THIS]
// write grid to file for later visualization :)
void write_grid_to_file(const Grid &g, const char *filename) {
  FILE *f = fopen(filename, "wb");
  if (!f) {
    fprintf(stderr, "ERROR: cannot open %s for writing\n", filename);
    return;
  }
  uint32_t W = g.W;
  uint32_t H = g.H;
  fwrite(&W, sizeof(uint32_t), 1, f);
  fwrite(&H, sizeof(uint32_t), 1, f);
  for (size_t i = 0; i < W * H; i++)
    fwrite(&g.alive[i], sizeof(uint8_t), 1, f);
  for (size_t i = 0; i < W * H; i++)
    fwrite(&g.hue[i], sizeof(float), 1, f);
  fclose(f);
}

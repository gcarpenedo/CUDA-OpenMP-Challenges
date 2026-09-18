#ifndef GRID_H
#define GRID_H

#include <vector>

#define MAX_STEPS 1000000
#define NUM_GROUPS 7
#define COLOR_VARIATION 0.08f
#define CLUSTER_FILL_DENSITY 0.55f
#define CLUSTER_RADIUS_FACTOR 0.08f
#define CLUSTER_MIN_RADIUS 3

// toggle this to save time when profiling
// be wary: this also disables the results check
#define DISABLE_SEQUENTIAL true

// if you happen to need extra information in the grid, you can modify this object, however,
// only add, do not remove anything, as to preserve compatibility with the sequential version!
//
// notes about the use of C++ here:
// - Grid is a C++ object owning two heap-allocated arrays
// - copying or assigning a Grid (Grid B = A; or B = A;) performs a deep copy of both vectors
// - passing Grid by value also triggers this deep copy, thus always pass by reference
// => this is different from C: the struct looks cheap, but copies are expensive
struct Grid {
  int W, H;
  std::vector<unsigned char> alive; // size: W*H
  std::vector<float> hue; // size: W*H
  // constructor: allocates and owns two dynamic arrays via std::vector
  Grid(int w, int h) : W(w), H(h), alive(w*h,0), hue(w*h,0.0f) {}
};

struct Grid_better{
  int W, H;
  std::vector<unsigned char> alive; // size: W*H
  std::vector<float> hue; // size: W*H
  std::vector<unsigned char> alive_neighbors; // counts alive neighbours
  std::vector<unsigned char> flags; // generic per-cell flag buffer

  // constructor: allocates and owns two dynamic arrays via std::vector
  Grid_better(int w, int h) : W(w), H(h), alive(w*h,0), hue(w*h,0.0f), alive_neighbors(w*h, 0), flags(w*h, 0) {}
};

#endif // GRID_H

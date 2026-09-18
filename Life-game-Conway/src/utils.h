#ifndef UTILS_H
#define UTILS_H

#include "grid.h"

// hue averaging using circular mean around the unit circle
float hue_average(const float *h, int n);

// more efficient function
float hue_avg_better(const float *h, int n);

// random hue selection
float pick_group_hue(int group_id);

// grid initialization with a few clusters
void initialize_grid(Grid &g);
void initialize_grid(Grid_better &g);

// compares two grids
bool compare_grids(const Grid &a, const Grid &b);
bool compare_grids(const Grid &a, const Grid_better &b);
bool compare_grids(const Grid_better &a, const Grid &b);

// write grid to file for later visualization
void write_grid_to_file(const Grid &g, const char *filename);

#endif // UTILS_H

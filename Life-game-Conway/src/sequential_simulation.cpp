#include "simulation.h"
#include "rules.h"
#include "utils.h"

// advance the simulation by one step; return the count of changed cells
int evolve_step(const Grid &cur, Grid &next) {
  int W = cur.W, H = cur.H;
  int changes = 0;

  // iterate over all cells
  for (int x = 0; x < W; x++) {
    for (int y = 0; y < H; y++) {
    int idx = y*W + x;
    unsigned char alive = cur.alive[idx];

    int alive_neighbors = 0;
    float parent_hues[8];

    // count alive moore neighbors and collect their hues
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        if (dx == 0 && dy == 0) continue;
        // wrap around the grid (torus)
        int xx = (x + dx + W) % W;
        int yy = (y + dy + H) % H;
        int nidx = yy*W + xx;
        if (cur.alive[nidx]) {
          parent_hues[alive_neighbors] = cur.hue[nidx];
          alive_neighbors++;
        }
      }
    }

    unsigned char new_alive = alive;

    if (!alive) {
      if (birth_rule(alive_neighbors)) {
        new_alive = 1;
        next.hue[idx] = hue_average(parent_hues, alive_neighbors);
      }
    } else {
      // you are free to skip this check, here it was kept for sake of completeness
      if (survive_rule(alive_neighbors)) {
        new_alive = 1;
        next.hue[idx] = cur.hue[idx];
      } else {
        new_alive = 0;
      }
    }

    next.alive[idx] = new_alive;
    if (new_alive != alive)
      changes++;
    }
  }
  return changes;
}

// sequential simulation entry point
void simulate_sequential(Grid &g) {
  // it's hard to perform updates in place, we ping-pong between grid copies
  Grid tmp(g.W, g.H);

  for (long step = 0; step < MAX_STEPS; step++) {
    // one step at a time
    // note: fully overwrites tmp with the next state
    int changes = evolve_step(g, tmp);

    // swap g <-> tmp
    g.alive.swap(tmp.alive);
    g.hue.swap(tmp.hue);

    if (changes == 0) break;
  }
}

// === DO NOT CHANGE ANYTHING ABOVE THIS COMMENT ===
// =================================================
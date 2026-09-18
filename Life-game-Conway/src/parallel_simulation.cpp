#include "simulation.h"
#include "rules.h"
#include "utils.h"

void simulate_parallel(Grid &g) {

  // use the given matrix + a tmp one, they will
  // ping-pong switched by switching the pointer
  Grid tmp(g.W, g.H);

  // other useful variables
  int changes = 0;
  int W = g.W, H = g.H;
  int alive_neighbors = 0;
  float parent_hues[8];
  unsigned char alive = 0;

  // until stop
  for (long step = 0; step < MAX_STEPS; ++step){

  // zero the count
    changes = 0;

  // set condtion for elements to iterate on

  // iterate on all the elements selected
    #pragma omp parallel private(alive_neighbors, parent_hues, alive)
    #pragma omp for schedule(guided) collapse(2) 
    for(int y = 0; y < H; ++y){
      for(int x = 0; x < W; ++x){

        // some self-notions
        int idx = y*W + x;
        alive = g.alive[idx];
        alive_neighbors = 0;

        // if alive skip
        if(alive){
          tmp.alive[idx] = 1;
          tmp.hue[idx] = g.hue[idx];
          continue;
        }
        
        // check neighbours
        for (int dx = -1; dx <= 1; dx++) {
          for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;

            // wrap around the grid (torus)
            int xx = (x + dx + W) % W;
            int yy = (y + dy + H) % H;
            int nidx = yy*W + xx;

            if (g.alive[nidx]) {
              parent_hues[alive_neighbors] = g.hue[nidx];
              alive_neighbors++;
            }
          }
        }


        // if passes rule
        if (birth_rule(alive_neighbors)){

          // activate on the other grid
          tmp.alive[idx] = 1;
          
          // compute the color
          tmp.hue[idx] = hue_avg_better(parent_hues, alive_neighbors);

          changes++;
        }

      }
    }

    // if count still zero break loop
    if (changes == 0) break;
    
    // swap grids
    g.alive.swap(tmp.alive);
    g.hue.swap(tmp.hue);
  }
}

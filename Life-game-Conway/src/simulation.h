#ifndef SIMULATION_H
#define SIMULATION_H

#include "grid.h"

// advance the simulation by one step; return the count of changed cells
int evolve_step(const Grid &cur, Grid &next);

// sequential simulation entry point
void simulate_sequential(Grid &g);

// parallel simulation entry point
void simulate_parallel(Grid &g);

// parallel simulation with death list
void simulate_parallel2(Grid &g);

// parallel simulation experimental 3 (uses Grid_better for neighbor counts)
void simulate_parallel3(Grid_better &g);

// parallel simulation experimental 4
void simulate_parallel4(Grid_better &g);

#endif // SIMULATION_H

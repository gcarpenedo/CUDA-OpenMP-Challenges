#include "simulation.h"
#include "rules.h"
#include "utils.h"
#include <algorithm>
#include <omp.h>

/*
Core Ideas of the Algorithm:
- Iterate only on the death cells.
- Each thread scans a chunk of iterations, saves new born cells and hues, or 
  death cells in thread-local buffers.
- After a barrier, threads update the grid, and rebuild the dead-cell
  list for the next pass.
*/

//Just on dead cells, buffer births locally
void simulate_parallel2(Grid &g) {
  const int W = g.W;
  const int H = g.H;
  const int total_cells = W * H;

  // Number of OpenMP threads captured once to size per-thread buffers deterministically.
  int T_NUMBER = omp_get_max_threads();

  // Per-thread buckets of dead cells currently under evaluation.
  std::vector<std::vector<int>> death_vec(T_NUMBER);

  // Dead cells discovered during a generation and scheduled for the next pass.
  std::vector<std::vector<int>> next_death_vec(T_NUMBER);

  // Thread-local indices/hues for births to commit after the barrier.
  std::vector<std::vector<int>> thread_birth_idx_buffers(T_NUMBER);
  std::vector<std::vector<float>> thread_birth_hue_buffers(T_NUMBER);

  // Shared offsets used when flattening thread-local buckets.
  std::vector<int> offsets(T_NUMBER);

  const int DEAD_RESERVE = total_cells / T_NUMBER;
  const int BIRTH_RESERVE = T_NUMBER*8;

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < T_NUMBER; ++i) {
    death_vec[i].reserve(DEAD_RESERVE);
    next_death_vec[i].reserve(DEAD_RESERVE);
    thread_birth_idx_buffers[i].reserve(BIRTH_RESERVE);
    thread_birth_hue_buffers[i].reserve(BIRTH_RESERVE);
  }

  // Get the list of initial dead cells, in parallel
  #pragma omp parallel
  {
    int t_id = omp_get_thread_num();
    auto &bucket = death_vec[t_id];
    bucket.clear();
    int idx;

    #pragma omp for schedule(static) collapse(2)
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        idx = y * W + x;
        if (!g.alive[idx]) {
          bucket.push_back(idx);
        }
      }
    }
  }

  // vector for death cells, flat so that it may be accessed in parallel
  std::vector<int> flat_deaths;
  
  // obtain the offsets via prefix sum
  int total = 0;
  for (int i = 0; i < T_NUMBER; ++i) {
    offsets[i] = total;
    total += static_cast<int>(death_vec[i].size());
  }
  flat_deaths.resize(total);

  // fill the flat deaths vector, in parallel
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < T_NUMBER; ++i) {
    int off = offsets[i];
    const auto &bucket = death_vec[i];
    for (size_t j = 0; j < bucket.size(); ++j){
      flat_deaths[off + static_cast<int>(j)] = bucket[j];
    }
  }
  

  // time-step, repeat until done (break is called)
  for (long step = 0; step < MAX_STEPS; ++step) {
    int changes = 0;

    // at every time-step sum via reduction the changes made by the threads, if 0 break
    #pragma omp parallel reduction(+:changes)
    {
      // clean the vectors used later
      int t_id = omp_get_thread_num();
      auto &local_dead = next_death_vec[t_id];
      auto &local_birth_idx = thread_birth_idx_buffers[t_id];
      auto &local_birth_hue = thread_birth_hue_buffers[t_id];
      local_dead.clear();
      local_birth_idx.clear();
      local_birth_hue.clear();

      // iterate on the death elements, guided distribution of works best
      #pragma omp for schedule(guided)
      for (int idx = 0; idx < static_cast<int>(flat_deaths.size()); ++idx) {

        // info about the current cell
        const int cell = flat_deaths[idx];
        const bool alive = g.alive[cell];
        int alive_neighbors = 0;
        float parent_hues[8];
        const int x = cell % W;
        const int y = cell / W;

        // Evaluate the Moore neighborhood (expensive!)
        for (int dx = -1; dx <= 1; ++dx) {
          for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0)
              continue;

            int xx = (x + dx + W) % W;
            int yy = (y + dy + H) % H;
            int nidx = yy * W + xx;

            if (g.alive[nidx]) {
              parent_hues[alive_neighbors] = g.hue[nidx];
              alive_neighbors++;
            }
          }
        }
        
        // check if the cell should be born
        if (birth_rule(alive_neighbors)) {
          
          // need to save thread-locally the updates, so that the time-step stillness
          // logic will not be affected (update simultaneously all the time-step)
          float new_hue = hue_avg_better(parent_hues, alive_neighbors);
          local_birth_idx.push_back(cell);
          local_birth_hue.push_back(new_hue);
          changes++;

        } else if (!alive) {

          // need to be checked again next time-step (expensive to check them all!)
          local_dead.push_back(cell);
        }
      }

      // Synchronize before touching shared grid state.
      #pragma omp barrier

      // every thread has its vectors, so it's already paralllelized
      for (size_t i = 0; i < local_birth_idx.size(); ++i) {
        int cell_idx = local_birth_idx[i];
        g.alive[cell_idx] = 1;
        g.hue[cell_idx] = local_birth_hue[i];
      }
    }

    // stop criteria
    if (changes == 0)
      break;

    // seerially compute the offset in the flat vector
    int total = 0;
    for (int i = 0; i < T_NUMBER; ++i) {
      offsets[i] = total;
      total += static_cast<int>(next_death_vec[i].size());
    }
    flat_deaths.resize(total);

    // merge the vectors (parallely)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < T_NUMBER; ++i) {
      int off = offsets[i];
      const auto &bucket = next_death_vec[i];
      for (size_t j = 0; j < bucket.size(); ++j)
        flat_deaths[off + static_cast<int>(j)] = bucket[j];
    }

    death_vec.swap(next_death_vec);
  }
}

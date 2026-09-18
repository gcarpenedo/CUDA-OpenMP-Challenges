#include "simulation.h"
#include "rules.h"
#include "utils.h"
#include <algorithm>
#include <omp.h>

// Core Algorithm Ideas:
// - Iterate only on cells that may become alive in the current time-step
// - Thread-local buckets minimize synchronization and improve parallelism
// - Usage of separate vectors instead of Array of Structures improve efficiency 
//   (Structures of Arrays would require to be a vector to be parallelized, thus another Array of Structures)
// - Increment surrounding cell's alive neighbor counting upon cell birth instead of 
//    calculating the value at every time-step

// Update neighbor counts and enqueue dead neighbors without duplicates.
static inline void process_alive_cell(int cell_idx,
                                      int H,
                                      int W,
                                      std::vector<unsigned char> &counts,
                                      const std::vector<unsigned char> &alive,
                                      std::vector<unsigned char> &marks,
                                      std::vector<int> &bucket) {
  const int x = cell_idx % W;
  const int y = cell_idx / W;
  
  for (int dy = -1; dy <= 1; ++dy) {
    int yy = y + dy;
    if (yy < 0) yy += H;
    else if (yy >= H) yy -= H;
    int y_offset = yy * W;

    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0)
        continue;
      
      int xx = x + dx;
      if (xx < 0) xx += W;
      else if (xx >= W) xx -= W;
      
      int nidx = y_offset + xx;

      // atomic count of alive neighbors
      #pragma omp atomic
      counts[nidx] += 1;

      if (alive[nidx])
        continue;

      unsigned char prev = 0;
      // capture allows single atomic operation for "read + write" 
      #pragma omp atomic capture
      {
        prev = marks[nidx];
        marks[nidx] = 1;
      }

      if (!prev)  // if no other thread is trying to do the same
        bucket.push_back(nidx);
    }
  }
}


void simulate_parallel3(Grid_better &g){
  const int W = g.W;
  const int H = g.H;
  const int tot_cells = W*H;

  int THREADS = omp_get_max_threads();

  // the vectors that will be used
  std::vector<std::vector<int>> curr_death_vec(THREADS);
  std::vector<std::vector<int>> next_death_vec(THREADS);
  std::vector<std::vector<int>> birth_vec(THREADS);
  std::vector<std::vector<float>> hue_vec(THREADS);
  std::vector<int> offsets(THREADS);
  std::vector<int> flat_vec_dead;

  const int RES_D = tot_cells / THREADS;
  const int RES_B = std::max(W, H) / THREADS * 8;  

  // prepare manually the buffers for the first round 
  #pragma omp parallel 
  {
    int t_id = omp_get_thread_num();

    // initialize the thread-local vectors
    curr_death_vec[t_id].reserve(RES_D);
    next_death_vec[t_id].reserve(RES_B);
    birth_vec[t_id].reserve(RES_B);
    hue_vec[t_id].reserve(RES_B);

    auto &bucket = curr_death_vec[t_id];
    bucket.clear();
    curr_death_vec[t_id].clear();
    next_death_vec[t_id].clear();
    birth_vec[t_id].clear();
    hue_vec[t_id].clear();
     
    // fill the thread-local buckets
    #pragma omp for schedule(static) collapse(2)
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        int idx = y * W + x; 

        if (g.alive[idx]) {
          process_alive_cell(idx, H, W,
                             g.alive_neighbors,
                             g.alive,
                             g.flags,
                             bucket);
        }
      }
    }
  } // end of parallel section

  // prefix sum for offsets
  int total = 0;
  for (int i = 0; i < THREADS; ++i){
    offsets[i] = total;
    total += static_cast<int>(curr_death_vec[i].size());
  }
  flat_vec_dead.resize(total);

  // gather the local vecotrs into a unique flat vector, parallely
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < THREADS; ++i) {
    int off = offsets[i];
    auto &bucket = curr_death_vec[i];

    for (size_t j = 0; j < bucket.size(); ++j) {
      int cell_idx = bucket[j];
      flat_vec_dead[off + static_cast<int>(j)] = cell_idx;
      g.flags[cell_idx] = 0;
    }
    bucket.clear();
  }
  
  // begin the time-step part
  for (long step = 0; step < MAX_STEPS; ++step){
    
    int changes = 0;

    // work is spread for parallelization, sum of changes with reduction
    #pragma omp parallel reduction(+ : changes)
    {
      int t_id = omp_get_thread_num();

      // local pointers to the vectors
      auto &local_next = next_death_vec[t_id];
      auto &local_birth = birth_vec[t_id];
      auto &local_hue = hue_vec[t_id];

      local_next.clear();
      local_birth.clear();
      local_hue.clear();

      // iterate on cells that may become alive this time-step
      #pragma omp for schedule(static) 
      for (int idx = 0; idx < static_cast<int>(flat_vec_dead.size()); ++idx){
        const int cell = flat_vec_dead[idx];

        if (g.alive[cell]) continue; // there should be none

        int alive_neighbors = g.alive_neighbors[cell];
        const int x = cell % W;
        const int y = cell / W;

        if (birth_rule(alive_neighbors)){
          
          // save parent hues
          float parent_hues[8] = {0.f};
          int hue_count = 0;
          
          for (int dy = -1; dy <= 1; ++dy) {
            int yy = y + dy;
            if (yy < 0) yy += H;
            else if (yy >= H) yy -= H;
            int y_offset = yy * W;

            for (int dx = -1; dx <= 1; ++dx) {
              if (dx == 0 && dy == 0) continue;
              
              int xx = x + dx;
              if (xx < 0) xx += W;
              else if (xx >= W) xx -= W;
              
              int nidx = y_offset + xx;
              if (g.alive[nidx])
                parent_hues[hue_count++] = g.hue[nidx];
            }
          }
          
          // may use the regular hue calculation if this doesn't work
          // float new_hue = hue_average(parent_hues, hue_count);
          float new_hue = hue_avg_better(parent_hues, hue_count);
          local_birth.push_back(cell);
          local_hue.push_back(new_hue);
          changes++;
        }
      } // implicit barrier

      // already in a parallel section
      for(size_t i = 0; i < local_birth.size(); ++i){
        int cell_idx = local_birth[i];
        g.alive[cell_idx] = 1;
        g.hue[cell_idx] = local_hue[i];
        g.flags[cell_idx] = 0;

        process_alive_cell(cell_idx,
                           H,
                           W,
                           g.alive_neighbors,
                           g.alive,
                           g.flags,
                           local_next);
      }
    }

    if (changes == 0)
      break;

    // gather the buckets into the flat_vector
    int total = 0;
    for (int i = 0; i < THREADS; ++i) {
      offsets[i] = total;
      total += static_cast<int>(next_death_vec[i].size());
    }
    flat_vec_dead.resize(total);

    // prepare the flat vector for the next time-step
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < THREADS; ++i) {
      int off = offsets[i];
      auto &bucket = next_death_vec[i];
      for (size_t j = 0; j < bucket.size(); ++j) {
        int cell_idx = bucket[j];
        flat_vec_dead[off + static_cast<int>(j)] = cell_idx;
        g.flags[cell_idx] = 0;
      }
      bucket.clear();
    }
  }
}
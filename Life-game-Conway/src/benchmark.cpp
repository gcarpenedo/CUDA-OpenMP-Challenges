#include "grid.h"
#include "utils.h"
#include "simulation.h"
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

void run_benchmarks(int seed) {
    auto benchmark_once = [](int Wb, int Hb, int seedb) {
      srand(seedb);
      Grid base(Wb, Hb);
      Grid_better exp(Wb, Hb);
      initialize_grid(base);
      
      // Manual copy for Grid_better
      std::copy(base.alive.begin(), base.alive.end(), exp.alive.begin());
      std::copy(base.hue.begin(), base.hue.end(), exp.hue.begin());
      // alive_neighbors and flags are already 0-initialized by constructor

      const int runs = 5;
      double tp1_sum = 0.0, tp2_sum = 0.0, tp3_sum = 0.0, tp4_sum = 0.0;

      bool compared = false;
      for (int i = 0; i < runs; ++i) {
        Grid g1 = base, g2 = base;
        Grid_better g3 = exp, g4 = exp;

        double s1 = omp_get_wtime();
        simulate_parallel(g1);
        double e1 = omp_get_wtime();
        tp1_sum += (e1 - s1);

        double s2 = omp_get_wtime();
        simulate_parallel2(g2);
        double e2 = omp_get_wtime();
        tp2_sum += (e2 - s2);

        double s3 = omp_get_wtime();
        simulate_parallel3(g3);
        double e3 = omp_get_wtime();
        tp3_sum += (e3 - s3);

        double s4 = omp_get_wtime();
        simulate_parallel4(g4);
        double e4 = omp_get_wtime();
        tp4_sum += (e4 - s4);

        if (!compared) {
          // single comparison against method1 (parallel std) on first run
          bool eq2 = compare_grids(g1, g2);
          bool eq3 = compare_grids(g1, g3);
          bool eq4 = compare_grids(g1, g4);
          printf("[W=%d H=%d] One-run check vs Method1: M2:%s M3:%s M4:%s\n",
               Wb, Hb,
               eq2 ? "PASS" : "FAIL",
               eq3 ? "PASS" : "FAIL",
               eq4 ? "PASS" : "FAIL");
          compared = true;
        }
      }

      printf("[W=%d H=%d] Avg Parallel1: %.6f s | Parallel2: %.6f s | Parallel3: %.6f s | Parallel4: %.6f s\n",
           Wb, Hb,
           tp1_sum / runs,
           tp2_sum / runs,
           tp3_sum / runs,
           tp4_sum / runs);
    };

    // Run requested benchmark sizes using provided seed
    benchmark_once(150, 150, seed);
    benchmark_once(800, 800, seed);
    benchmark_once(1500, 1500, seed);
    benchmark_once(2000, 2000, seed);
}

int main() {
    run_benchmarks(67);
    return 0;
}

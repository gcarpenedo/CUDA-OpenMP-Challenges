#include <cstdio>
#include <cstdlib>
#include <omp.h>

#include "grid.h"
#include "rules.h"
#include "utils.h"
#include "simulation.h"

// === DO NOT CHANGE ANYTHING ABOVE THIS COMMENT ===
// =================================================

// if you need any additional standard library up to C11 or C++17, add it here

// if you need any additional define, add it here

// =================================================
// === DO NOT CHANGE ANYTHING BELOW THIS COMMENT ===

int main(int argc, char **argv) {
    omp_set_nested(true); // just in case, enable nested parallelism

    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s <grid-width> <grid-height> <seed> <[opt]-output-filename>\n", argv[0]);
        return 1;
    }

    int W = atoi(argv[1]);
    int H = atoi(argv[2]);
    int seed = atoi(argv[3]);
    srand(seed);

    Grid gs(W,H), gp1(W,H), gp2(W,H);
    Grid_better gp3(W,H), gp4(W,H);
    initialize_grid(gs);
    // copy the initial state
    gp1 = gs;
    gp2 = gs;

    // Manual copy for Grid_better (else would use another seed)
    std::copy(gs.alive.begin(), gs.alive.end(), gp3.alive.begin());
    std::copy(gs.hue.begin(), gs.hue.end(), gp3.hue.begin());

    std::copy(gs.alive.begin(), gs.alive.end(), gp4.alive.begin());
    std::copy(gs.hue.begin(), gs.hue.end(), gp4.hue.begin());
    
    // === DO NOT CHANGE ANYTHING ABOVE THIS COMMENT ===
    // =================================================

    // if you need to initialize additional things, do that here!

    // =================================================
    // === DO NOT CHANGE ANYTHING BELOW THIS COMMENT ===

    // sequential
    #if !DISABLE_SEQUENTIAL
    double t1 = omp_get_wtime();
    simulate_sequential(gs);
    double t2 = omp_get_wtime();
    #endif

    // parallel std version
    double t3 = omp_get_wtime();
    simulate_parallel(gp1);
    double t4 = omp_get_wtime();

    // parallel with death buffer
    double t5 = omp_get_wtime();
    simulate_parallel2(gp2);
    double t6 = omp_get_wtime();

    // parallel experimental
    double t7 = omp_get_wtime();
    simulate_parallel3(gp3);
    double t8 = omp_get_wtime();

    // parallel experimental 4
    double t9 = omp_get_wtime();
    simulate_parallel4(gp4);
    double t10 = omp_get_wtime();

    #if !DISABLE_SEQUENTIAL
    printf("Sequent. time: %.6f s\n", t2 - t1);
    #endif
    printf("Parallel time: %.6f s\n", t4 - t3);
    printf("Parallel2 time: %.6f s\n", t6 - t5);
    printf("Parallel3 time: %.6f s\n", t8 - t7);
    printf("Parallel4 time: %.6f s\n", t10 - t9);

    bool equal = compare_grids(gp1, gp2);
    printf("Results check on Method 2: %s\n", equal ? "PASS" : "FAIL");
    equal = compare_grids(gp1, gp3);
    printf("Results check on Method 3: %s\n", equal ? "PASS" : "FAIL");
    equal = compare_grids(gp1, gp4);
    printf("Results check on Method 4: %s\n", equal ? "PASS" : "FAIL");

    // === DO NOT CHANGE ANYTHING ABOVE THIS COMMENT ===
    // =================================================

    // if you need more logging, put it here!

    // =================================================
    // === DO NOT CHANGE ANYTHING BELOW THIS COMMENT ===

    if (argc == 5) {
      write_grid_to_file(gp1, (char *)argv[4]);
      printf("Results written to %s\n", (char *)argv[4]);
    }

    return 0;
}

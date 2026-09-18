# Parallel Computing Challenges

Implementations for parallel computing assignments from the Parallel Computing course at Politecnico di Milano. The repository contains one CUDA project and one OpenMP project.

## Projects

### 1. CUDA-GeMM: Batched Matrix Multiplication

A high-performance CUDA implementation of batched General Matrix Multiplication (GeMM) with advanced optimization techniques.

**Location:** `CUDA-GeMM/`

**Description:**
- Implements batched matrix multiplication: `P = M × N` for multiple matrix pairs
- Matrix M is shared across all batches, while N and P are batch-specific
- Utilizes GPU acceleration with CUDA for massive parallelization

**Key Features:**
- **Tiling optimization** with configurable tile size (64×64)
- **Micro-tiling** strategy for better register utilization (2×16 micro-tiles per thread)
- **Shared memory** usage for reduced global memory access
- **Thread block configuration** optimized for GPU architecture (32×4 threads)
- Handles arbitrary matrix dimensions with boundary checking

**Technical Details:**
- Tile size: 64×64
- Block dimensions: 32×4 threads
- Micro-tile per thread: 2×16 elements
- Uses shared memory for M and N matrices during computation

**Build and run:**
```bash
cd CUDA-GeMM/src
make
./gemm <m> <k> <n> <batch> <seed>
```

The build requires the NVIDIA CUDA Toolkit and a GPU compatible with the architecture configured in `src/Makefile` (`sm_75`). The executable writes the generated matrices and parameters to `M.bin`, `N.bin`, `P_gpu.bin`, and `params.txt` in the current directory.

**Parameters:**
- `m`: Number of rows in M and P
- `k`: Number of columns in M, rows in N
- `n`: Number of columns in N and P
- `batch`: Number of matrix pairs to process
- `seed`: Random seed for initialization

---

### 2. Life-game-Conway: Conway's Game of Life with OpenMP

A parallel implementation of a Conway-style cellular automaton using OpenMP, with extended features including cell coloring based on parent lineage.

**Location:** `Life-game-Conway/`

**Description:**
- Implements the B368/S012345678 cellular automaton with both sequential and parallel versions
- Enhanced with HSV color tracking to visualize cell lineage
- Supports both sequential and parallel execution with performance comparison

**Rules:**
- **Birth rule:** A dead cell with 3, 6, or 8 alive neighbors becomes alive
- **Survival rule:** A living cell survives with any number of alive neighbors from 0 to 8
- **Color inheritance:** New cells inherit averaged hue from their parent cells
- **Boundary conditions:** Toroidal grid (wraps around edges)

**Key Features:**
- Grid-based simulation with configurable width and height
- Sequential baseline implementation
- Parallel implementation using OpenMP (to be optimized)
- Performance benchmarking between sequential and parallel versions
- Optional output to file for visualization

**Build and run:**
```bash
cd Life-game-Conway
mkdir -p build results && cd build
cmake ..
cmake --build .
./chal2 <grid-width> <grid-height> <seed> [../results/output-filename]
```

The project requires CMake, a C++17 compiler, and OpenMP. The CMake configuration builds both `chal2` and `benchmark`.

**Parameters:**
- `grid-width`: Width of the simulation grid
- `grid-height`: Height of the simulation grid
- `seed`: Random seed for initial state generation
- `output-filename`: (Optional) binary file to save the final grid state

**Performance Metrics:**
- Measures execution time for sequential and parallel versions
- Reports speedup factor
- Validates correctness by comparing results

To run the benchmark executable:

```bash
./benchmark
```

The optional output file uses a binary format containing the grid width, height, alive-cell data, and hue data. The plotting helper is available at `Life-game-Conway/tools/plotter.py` and can be run with:

```bash
python3 ../tools/plotter.py ../results/final_grid.bin
```
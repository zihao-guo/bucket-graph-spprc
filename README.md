# bucket-graph-spprc

[![CI](https://github.com/spoorendonk/bucket-graph-spprc/actions/workflows/ci.yml/badge.svg)](https://github.com/spoorendonk/bucket-graph-spprc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)

Header-only C++23 template library implementing the bucket graph labeling algorithm for the Shortest Path Problem with Resource Constraints (SPPRC) and vehicle routing variants. Based on [Sadykov, Uchoa & Pessoa (2021)](https://doi.org/10.1287/trsc.2020.0985) with extensions from [Sadykov et al. (2026)](https://inria.hal.science/hal-05486295v2).

## Key Features

- **Generic Resource concept** — 7-function compile-time interface
- **5 built-in resources** — Standard (time/capacity), NgPath, R1C (rank-1 cuts), CumulativeCost, PickupDelivery
- **Mono and bidirectional labeling** with across-arc concatenation
- **Multi-stage heuristics** — Heuristic1 → Heuristic2 → Exact → Enumerate
- **Bucket fixing and arc elimination** with completion-bound pruning
- **SoA label storage** with SIMD-accelerated dominance checks
- **2D bucketing** for problems with two main resources

## Quick Start

Requires GCC 14+ and CMake 3.25+.

```bash
cmake -B build -DCMAKE_CXX_COMPILER=g++-14
cmake --build build

# One-time: fetch benchmark instances
benchmarks/fetch_instances.sh

# Solve an SPPRC instance
./build/bgspprc-solve --stats --timing benchmarks/instances/spprclib/A-n54-k7-149.sppcc
```

Sample output:

```
bgspprc 1e749ce
bgspprc: executor=StdThread  bidir=parallel
A-n54-k7-149   sppcc  n=55  arcs=2862  theta=-1e-06  cost=-56718.000  paths=1  18.1ms
  0 34 41 25 47 25 47 25 41 34 54
  n_buckets=550  n_labels_created=113210  n_dominance_checks=128977  n_non_dominated=12059
  n_dominance_checks_fw=53407  n_dominance_checks_bw=75570
  n_non_dominated_fw=4854  n_non_dominated_bw=7205
  n_fixed_buckets=0  n_eliminated_arcs=0  label_state_bytes=1
  timing:  fw=5.721ms  bw=9.660ms  completion=0.987ms  concat=2.291ms  paths=1.109ms  sum=19.768ms
```

Supported instance formats: `.sppcc` (SPPCC), `.vrp` (Roberti VRPTW), `.graph` (Solomon RCSPP).
Full flag reference in [CLI Reference](#cli-reference) below.

## Benchmarks

```bash
# Fetch benchmark instances (once)
benchmarks/fetch_instances.sh

# Run benchmarks
benchmarks/run_benchmarks.sh
```

### Results at a glance

Headline numbers from the committed CSVs, 120 s timeout per instance. Full
tables, methodology, and reproducer one-liners in
[`benchmarks/README.md`](benchmarks/README.md#results).

Plato-style columns: `scaled (s)` = shifted geomean, `unscaled (s)` =
arithmetic mean, `solved` = #instances finished within the 120 s timeout
(TL substitutes as 120 s in both means).

**Pathwyse comparison (sppcc + vrp)** — bgspprc `para_bidir` vs patched
Pathwyse, both in pure-ng mode. sgm shift = 1 s.

| set      | ng | solver   | scaled (s) | unscaled (s) | solved |
|----------|---:|----------|-----------:|-------------:|-------:|
| spprclib |  8 | bgspprc  |       0.89 |         6.95 |  44/45 |
| spprclib |  8 | pathwyse |       1.52 |        12.13 |  42/45 |
| spprclib | 16 | bgspprc  |       2.00 |        15.67 |  40/45 |
| spprclib | 16 | pathwyse |       4.28 |        24.29 |  38/45 |
| spprclib | 24 | bgspprc  |       5.68 |        26.35 |  38/45 |
| spprclib | 24 | pathwyse |      10.59 |        42.51 |  31/45 |
| roberti  |  8 | bgspprc  |       0.54 |         0.88 |  31/31 |
| roberti  |  8 | pathwyse |       2.33 |        10.00 |  30/31 |
| roberti  | 16 | bgspprc  |       3.09 |        14.86 |  28/31 |
| roberti  | 16 | pathwyse |       8.80 |        28.15 |  28/31 |
| roberti  | 24 | bgspprc  |      14.36 |        43.70 |  23/31 |
| roberti  | 24 | pathwyse |      26.82 |        62.02 |  19/31 |

**Paper comparison (rcspp)** — bgspprc `para_bidir` vs Petersen & Spoorendonk
2025 (arXiv:2511.01397) `all_s` column. sgm shift = 10 s.

| ng | solver   | scaled (s) | unscaled (s) | solved |
|---:|----------|-----------:|-------------:|-------:|
|  8 | bgspprc  |       3.67 |         5.70 |  56/56 |
|  8 | paper    |       0.33 |         0.41 |  56/56 |
| 16 | bgspprc  |       4.66 |         8.91 |  56/56 |
| 16 | paper    |       1.37 |         3.01 |  56/56 |
| 24 | bgspprc  |       6.10 |        15.53 |  52/56 |
| 24 | paper    |       2.89 |         9.12 |  53/56 |

Both solvers reach the same optimal reduced cost on `.sppcc`/`.vrp`
modulo cost-scale rounding (verified per-row in `comparison_pathwyse.csv`).
Pathwyse needs patches against upstream `d53c01b` —
see [`benchmarks/patches/`](benchmarks/patches/) (auto-applied by
`run_pathwyse.sh`).

## Use as a Library

Add via CMake FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(
    bgspprc
    GIT_REPOSITORY https://github.com/spoorendonk/bucket-graph-spprc.git
    GIT_TAG main)
FetchContent_MakeAvailable(bgspprc)

target_link_libraries(your_target PRIVATE bgspprc)
```

Minimal usage:

```cpp
#include <bgspprc/solver.h>

using namespace bgspprc;

// Set up problem data (caller owns all arrays)
ProblemView pv;
pv.n_vertices = N;
pv.source = 0;
pv.sink = N - 1;
pv.n_arcs = M;
pv.arc_from = from.data();
pv.arc_to = to.data();
pv.arc_base_cost = cost.data();
pv.n_resources = 1;
pv.arc_resource = &arc_resource_ptr;
pv.vertex_lb = &vertex_lb_ptr;
pv.vertex_ub = &vertex_ub_ptr;

// Create solver with no extra resources
Solver<EmptyPack> solver(pv, EmptyPack{},
    {.bucket_steps = {10.0, 1.0}});
solver.set_stage(Stage::Exact);
solver.build();

auto paths = solver.solve();
for (auto& p : paths)
    printf("cost=%.2f  vertices=%zu\n", p.reduced_cost, p.vertices.size());
```

## Custom Resources

Implement the `Resource` concept to define custom resource types:

```cpp
struct MyResource {
    using State = double;

    bool symmetric() const;
    State init_state(Direction dir) const;
    std::pair<State, double> extend_along_arc(Direction dir, State s, int arc) const;
    std::pair<State, double> extend_to_vertex(Direction dir, State s, int vertex) const;
    double domination_cost(Direction dir, int vertex, State s1, State s2) const;
    double concatenation_cost(Symmetry sym, int vertex, State s_fw, State s_bw) const;
    double min_domination_cost() const;
};

// Bundle into a resource pack
using MyPack = ResourcePack<StandardResource, MyResource>;
Solver<MyPack> solver(pv, MyPack{std_res, my_res}, opts);
```

See [`examples/custom_resource.cpp`](examples/custom_resource.cpp) for a complete working example, [`include/bgspprc/resource.h`](include/bgspprc/resource.h) for the full concept definition, and [`include/bgspprc/resources/`](include/bgspprc/resources/) for built-in implementations.

## Examples

The [`examples/`](examples/) directory contains standalone programs, built by default at top level:

- **`basic_spprc.cpp`** — Solve a 5-vertex SPPRC with time windows
- **`custom_resource.cpp`** — Implement a custom capacity resource using the `Resource` concept

```bash
./build/examples/example_basic_spprc
./build/examples/example_custom_resource
```

## CLI Reference

```
Usage: bgspprc-solve [OPTIONS] <path>...

Arguments:
  <path>    Instance file or directory (recurse, detect type by extension)

Options:
  --version       Print build git hash and exit
  --mono          Use mono solver (default: bidir)
  --stage STAGE   heuristic1|heuristic2|exact (default: exact)
  --ng K          ng-neighborhood size (default: 0/off for sppcc/vrp;
                  from file or 8 for graph; 0 disables)
  --steps S1,S2   Bucket step sizes (default: per-type)
  --max-paths N   Number of paths to return (0=all, 1=best; default: 1)
  --theta T       Pricing threshold θ (default: -1e-6 for CG)
  --auto-steps    Use per-vertex auto-computed steps
  --stats         Print solve statistics after each instance
  --csv           Machine-readable CSV output
  --timing        Print phase timing breakdown
  --no-parallel   Use sequential executor (default: parallel)
  --no-parallel-bidir  Sequential fw/bw labeling
```

## Running Tests

```bash
# Run all tests (~195 unit tests)
ctest --test-dir build

# Run a specific test by name
./build/test_runner --test-case="Bucket construction"

# Run tests matching a pattern
./build/test_runner "*NgPath*"
```

## References

1. **Sadykov, Uchoa, Pessoa (2021)** — *A bucket graph-based labeling algorithm with application to vehicle routing*. Transportation Science, 55(1):4-28. DOI: [10.1287/trsc.2020.0985](https://doi.org/10.1287/trsc.2020.0985)

2. **Sadykov, Froger, Uchoa, Pessoa, Bulhoes, de Araujo (2026)** — *Bucket graph meta-solver for the resource constrained shortest path problem* (Meta-Solver). HAL: hal-05486295. <https://inria.hal.science/hal-05486295v2>

3. **Petersen, Spoorendonk (2025)** — *A parallel pull labelling algorithm for the resource constrained shortest path problem*. arXiv:2511.01397. <https://arxiv.org/abs/2511.01397>

4. **Pessoa, Sadykov, Uchoa, Vanderbeck (2020)** — *A generic exact solver for vehicle routing and related problems* (VRPSolver). Mathematical Programming, 183:483-523. DOI: [10.1007/s10107-020-01523-z](https://doi.org/10.1007/s10107-020-01523-z)

5. **Salani, Basso, Giuffrida (2024)** — *PathWyse: a flexible, open-source library for the resource constrained shortest path problem*. Optimization Methods and Software, 1–23. DOI: [10.1080/10556788.2023.2296978](https://doi.org/10.1080/10556788.2023.2296978)

6. **Seman, Munari, Bulhões, Camponogara (2024)** — *BALDES: A Branch-Cut-and-Price Bucket Graph Labeling Algorithm for Vehicle Routing*. GitHub: <https://github.com/lseman/baldes>

## Related Projects

- [**BALDES**](https://github.com/lseman/baldes) — Branch-Cut-and-Price bucket graph labeling for CVRP/VRPTW in C++ (Seman et al. 2024, ref [6]); BG2021 with R1C cuts and HGS-VRPTW heuristics
- [**PathWyse**](https://github.com/pathwyse/pathwyse) — Standard labeling + DSSR for RCSPP (Salani, Basso, Giuffrida 2024, ref [5])

## License

[MIT](LICENSE)

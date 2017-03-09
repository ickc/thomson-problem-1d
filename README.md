# Solving 1D "Thomson problem" with genetic algorithm

- particles with identical charge on an interval between -L to L
- length L is chosen to be 1
- Simulation is from 0 to L by symmetry
- actual number of particles is 2n

# Usage

Local:

```bash
make clean && make -j
thomson-problem-1d -h
```

On NERSC:

```bash
# compile (make sure the native flags are compile on the actual Cori node rather than login node)
sbatch ./compile-cori
# plot (output as CSV files)
sbatch ./scaling-hybrid
sbatch ./scaling-mpi
sbatch ./scaling-openmp
```

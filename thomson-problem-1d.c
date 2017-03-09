#include "mpi.h"
#include "omp.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef GETTIMEOFDAY
#include <sys/time.h> // For struct timeval, gettimeofday
#else
#include <time.h> // For struct timespec, clock_gettime, CLOCK_MONOTONIC
#endif

/* Solving 1D "Thomson problem" with genetic algorithm:

- particles with identical charge on an interval between -L to L
- length L is chosen to be 1
- Simulation is from 0 to L by symmetry
- actual number of particles is 2n */

// Utilities ///////////////////////////////////////////////////////////

double wall_time()
{
#ifdef GETTIMEOFDAY
    struct timeval t;
    gettimeofday(&t, NULL);
    return 1. * t.tv_sec + 1.e-6 * t.tv_usec;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1. * t.tv_sec + 1.e-9 * t.tv_nsec;
#endif
}

static inline void print_summary(char* filename, int n, int n_proc, int iterations, double seconds, double potential)
{
    /* print out summary
    append to existing file
    use this header:
    "OpenMP,MPI,n,iterations,t (s),V" */
    /* print all x values vs. the charge density lambda */
    if (!filename)
        return;
    FILE* fp;
    fp = fopen(filename, "a");
    fprintf(fp, "%d,%d,%d,%d,%f,%f\n", omp_get_max_threads(), n_proc, n, iterations, seconds, potential);
    fclose(fp);
}

static inline void print_all_x(char* filename, int n, double* x)
{
    /* print all x values vs. the charge density lambda
    the x value is chosen to be the middle point of each interval */
    if (!filename)
        return;
    FILE* fp;
    fp = fopen(filename, "w");

    double dx;

    fprintf(fp, "i,x,lambda\n");
    // i = 0 needed to be treated differently since the interval is between -x_0 to x_0
    // over n to normalize for the increase in n while keeping total Q constant
    double lambda = 1 / (2 * x[0]) / n;
    // TODO: impove x
    fprintf(fp, "%d,%f,%f\n", 0, 0, lambda);
    for (int i = 1; i < n; i++) {
        dx = x[i] - x[i - 1];
        lambda = 1 / dx / n;
        fprintf(fp, "%d,%f,%f\n", i, x[i] - dx / 2, lambda);
    }
    fclose(fp);
}

// initialize //////////////////////////////////////////////////////////

static inline double get_x_current(int n, double* x, int i, double ratio)
{
    /* i should never be the last, i.e. i != (n - 1)
    and the first one is actually in the middle so it should never went pass 0
    ratio is the ratio of x_range comparing to maximum possible dx */
    const double x_min = (i == 0) ? 0. : x[i - 1];
    const double x_max = x[i + 1];
    const double x_range = x_max - x_min;
    double x_current;
    // full range between x_min and x_max
    // do {
    //     my_random = drand48();
    //     x_current = x_min + my_random * x_range;
    // } while (my_random == 0 || x_current == x[i]);
    // only a narrow range around x[i]
    do {
        // the random number in parenthesis is in [-1, 1)
        x_current = x[i] + (2 * drand48() - 1) * x_range / ratio;
        // repeat when x_current fall out of range
    } while (x_current <= x_min || x_current >= x_max);
    // printf("%f\t%f\t%f\n", x_min, x_current, x_max); //debug
    return x_current;
}

static inline void init_particles(int n, double* x, double* global_potential, int n_proc, int rank)
{
    /* evenly spaced between -L to L, and x only holds the values between 0 to L
calculate the potential in this configuration by factoring out identical intervals in O(n)
V = \frac{1}{dx} \sum_{i = 1}^{N - 1}(\frac{N}{i} - 1), dx = \frac{L}{N-1}, N = 2n, L = 2 */
    double potential = 0;
    int N = 2 * n;
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < n; i++) {
            x[i] = (double)(2 * i + 1) / (N - 1);
        }
// each MPI process do their own sum
#pragma omp for reduction(+ : potential)
        for (int i = rank + 1; i < N; i += n_proc) {
            potential += (double)N / i - 1;
        }
    }
    MPI_Allreduce(&potential, &(*global_potential), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    *global_potential = *global_potential * (N - 1) / 2;
}

// Interactions ////////////////////////////////////////////////////////

static inline double _coulumb_self(double xi)
{
    /* Coulumb interaction between (i, -i) */
    return 1 / (2 * xi);
}

static inline double _coulumb(double xi, double xj)
{
    /* Coulumb interaction between (i, j), (i, -j), (-i, j), (-i, -j)
    assume r != 0 */
    const double rij = fabs(xi - xj);
    // r_{i,-j}
    const double rinj = xi + xj;
    const double Vij = 1 / rij;
    // include the charges at "-i" and "-j"
    const double Vinj = 1 / rinj;
    // the factor of 2 account for the fact they come in pairs
    return (2 * (Vij + Vinj));
}

static inline double get_potential_delta(int n, double* x, int i, double x_current, int n_proc, int rank)
{
    /* Calculate the change in potential energy in O(n) by
    subtracting the old contribution of potential between i & j
    add the new contribution at new position x_current */
    double global_potential_delta = 0;
    double potential_delta = 0;
    if (rank == 0) {
        potential_delta += _coulumb_self(x_current) - _coulumb_self(x[i]);
    }
// each MPI worker do their own sum
#pragma omp parallel for reduction(+ : potential_delta)
    for (int j = rank; j < n; j += n_proc) {
        if (j == i)
            continue;
        potential_delta += _coulumb(x_current, x[j]) - _coulumb(x[i], x[j]);
    }
    MPI_Allreduce(&potential_delta, &global_potential_delta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_potential_delta;
}

static inline double get_potential(int n, double* x)
{
    /* calculate the potential energy in the whole distribution in O(n^2) 
    **this one is not used** since the potential is initialized and potential_delta is added per iteration
    MPI is not implemented here
    for debug only */
    double potential = 0;
#pragma omp parallel reduction(+ : potential)
    {
#pragma omp for
        for (int i = 0; i < n; i++) {
            potential += _coulumb_self(x[i]);
        }
#pragma omp for
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                potential += _coulumb(x[i], x[j]);
            }
        }
    }
    return potential;
}

// Main ////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    // Setup MPI ///////////////////////////////////////////////////////

    int n_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get args ////////////////////////////////////////////////////////

    // default values for args
    int n = 10;
    int t = 10;
    // only needed by rank 0
    long int seed = 10;
    char* filename_summary = NULL;
    char* filename_x = NULL;
    char* filename_potential = NULL;

    if (rank == 0) {
        int opt;
        while ((opt = getopt(argc, argv, "hn:t:s:o:x:p:")) != -1) {
            switch (opt) {
            case 'n':
                n = (int)strtol(optarg, NULL, 0);
                break;
            case 't':
                t = (int)strtol(optarg, NULL, 0);
                break;
            case 's':
                seed = (int)strtol(optarg, NULL, 0);
                break;
            case 'o':
                filename_summary = optarg;
                break;
            case 'x':
                filename_x = optarg;
                break;
            case 'p':
                filename_potential = optarg;
                break;
            case 'h':
                printf("Usage:\t%s [-n]\n", argv[0]);
                printf("\t-h\thelp\n");
                printf("\t-n\tnumber of particles\n");
                printf("\t-t\tnumber of iterations\n");
                printf("\t-s\tseed of random numbers\n");
                printf("\t-o\toutput filename for summary\n");
                printf("\t-x\toutput filename for charge distribution\n");
                printf("\t-p\toutput filename for the potential per iteration\n");
                MPI_Finalize();
                return (0);
            default:
                fprintf(stderr, "%s:\tinvalid option -- %c\n", argv[0], opt);
                fprintf(stderr, "Try `%s -h' for more information.\n", argv[0]);
                MPI_Finalize();
                return (1);
            }
        }
        // for get_x_current
        srand48(seed);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Initialize //////////////////////////////////////////////////////

    // time it
    double seconds;
    if (rank == 0)
        seconds = -wall_time();

    // initialize x
    double* x = (double*)malloc(n * sizeof(double));
    // save communication cost by not broadcast this in MPI
    double potential = 0;
    init_particles(n, x, &potential, n_proc, rank);

    // print
    if (rank == 0) {
        printf("#OpenMP\t%d\n", omp_get_max_threads());
        printf("#MPI\t%d\n", n_proc);
        printf("n\t%d\n", n);
    }

    // prepare to enter the loop ///////////////////////////////////////

    FILE* file_potential;
    if (rank == 0 && filename_potential) {
        file_potential = fopen(filename_potential, "w");
        fprintf(file_potential, "iterations,potential\n");
    }
    double x_current;
    double potential_delta;
    // debug
    // int mutation_counter = 0;
    // int total_iteration;
    // double mutation_percent;
    // double mutation_ratio;
    //
    // for do loop only
    // bool mutated;

    // Main loop ///////////////////////////////////////////////////////

    int iterations = 0;
    int write_potential_interval = t / n;
    if (write_potential_interval == 0)
        write_potential_interval = 1;
    // do loop to stop at a certain criteria
    // do {
    //     mutated = false;
    //     iterations++;
    // for loop for fix iterations
    for (iterations; iterations < t; iterations++) {
        // mutate x[i]. Do not mutate the one on the right boundary (x = L)
        for (int i = 0; i < n - 1; i++) {
            if (rank == 0) {
                x_current = get_x_current(n, x, i, 200);
            }
            MPI_Bcast(&x_current, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // printf("%d\t%f\n", i, x_current); // debug
            potential_delta = get_potential_delta(n, x, i, x_current, n_proc, rank);
            if (potential_delta < 0) {
                x[i] = x_current;
                if (rank == 0)
                    potential += potential_delta;
                // for do loop only
                // mutated = true;
                // debug
                // mutation_counter++;
                // if (i == n - 2){
                //     total_iteration = (n - 1) * (iterations + 1);
                //     // mutation_percent = mutation_counter / total_iteration;
                //     mutation_ratio = (double)total_iteration / mutation_counter;
                //     printf("%d / %d = %f\n", total_iteration, mutation_counter, mutation_ratio);
                //     printf("Potential is %f\n", potential);
                // }
            }
            // debug
            // potential = get_potential(n, x) / n / n;
            // potential_delta = potential_delta / n / n;
            // printf("%d\t%f\t%f\t%f\n", i, potential, potential_delta, x_current);
        }
        if (rank == 0 && filename_potential && iterations % write_potential_interval == 0)
            fprintf(file_potential, "%d,%f\n", iterations, potential / n / n);
    }
    // } while (mutated);

    // Finalize ////////////////////////////////////////////////////////

    if (rank == 0) {
        seconds += wall_time();
        // print
        printf("Iterations\t%d\n", iterations);
        printf("Time\t%f\n", seconds);
        // normalize potential from charge density
        potential = potential / n / n;
        printf("Potential\t%f\n", potential);

        print_summary(filename_summary, n, n_proc, iterations, seconds, potential);
        print_all_x(filename_x, n, x);
        // close
        if (filename_potential) {
            // write the last entry before closing
            fprintf(file_potential, "%d,%f\n", iterations, potential);
            fclose(file_potential);
        }
    }

    free(x);
    MPI_Finalize();
    return 0;
}

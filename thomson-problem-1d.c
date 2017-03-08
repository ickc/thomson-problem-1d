#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "omp.h"

/* Solving 1D "Thomson problem" with genetic algorithm:

- particles with identical charge on an interval between -L to L
- length L is chosen to be 1
- Simulation is from 0 to L by symmetry
- actual number of particles is 2n */

static inline void init_particles(int n, double* x)
{
    /* evenly spaced between -L to L, and x only holds the values between 0 to L */
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = (double) (2 * i + 1) / (2 * n - 1);
    }
}

static inline double get_x_current(int n, double* x, int i, double ratio)
{
    /* i should never be the last, i.e. i != (n - 1) */
    double x_current;
    const double x_min = (i == 0) ? 0. : x[i - 1];
    const double x_max = x[i + 1];
    const double dx = x_max - x_min;
    double my_random;
    // full range between x_min and x_max
    // do {
    //     my_random = drand48();
    //     x_current = x_min + my_random * dx;
    // } while (my_random == 0 || x_current == x[i]);
    // only a narrow range around x[i]
    do {
        my_random = 2 * drand48() - 1;
        x_current = x[i] + my_random * dx / 2 / ratio;
    } while (x_current <= x_min || x_current >= x_max);
    // printf("%f\t%f\t%f\n", x_min, x_current, x_max); //debug
    return x_current;
}

static inline double _coulumb_self(double xi)
{
    /* Coulumb interaction between i and -i */
    return 1 / (2 * xi);
}

static inline double _coulumb(double xi, double xj)
{
    /* assume r != 0 */
    const double rij = abs(xi - xj);
    // r_{i,-j}
    const double rinj = xi + xj;
    const double Vij = 1 / rij;
    // include the charges at "-i" and "-j"
    const double Vinj = 1 / rinj;
    // the factor of 2 account for another pair in the "image" side
    return (2 * (Vij + Vinj));
}

static inline double get_potential_delta(int n, double* x, int i, double x_current)
{
    /* subtract the old contribution of potential between i & j
    add the new contribution at new position x_current */
    double potential_delta = _coulumb_self(x_current) - _coulumb_self(x[i]);
#pragma omp parallel for reduction( + : potential_delta )
    for (int j = 0; j < n; j++) {
        if (j == i)
            continue;
        potential_delta += _coulumb(x_current, x[j]) - _coulumb(x[i], x[j]);
    }
    return potential_delta;
}

static inline double get_potential(int n, double* x)
{
    double potential = 0;
#pragma omp parallel reduction( + : potential )
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

static inline void print_all(char* filename, int n, double* x)
{
    if (! filename)
        return;
    FILE* fp;
    fp = fopen (filename, "w");

    fprintf(fp, "i,x,lambda\n");
    // i = 0 needed to be treated differently since the interval is between -x_0 to x_0
    // over n to normalize for the increase in n while keeping total Q constant
    double lambda = 1 / (2 * x[0]) / n;
    fprintf(fp, "%d,%f,%f\n", 0, x[0], lambda);
    for (int i = 1; i < n; i++) {
        lambda = 1 / (x[i] - x[i - 1]) / n;
        fprintf(fp, "%d,%f,%f\n", i, x[i], lambda);
    }
    fclose(fp);
}

int main(int argc, char* argv[])
{
    // default values
    int n = 10;
    int t = 10;
    long int seed = 10;
    char* filename = NULL;

    // get arg
    int opt;
    while ((opt = getopt(argc, argv, "hn:t:o:s:")) != -1) {
        switch (opt) {
        case 'n':
            n = (int)strtol(optarg, NULL, 0);
            break;
        case 't':
            t = (int)strtol(optarg, NULL, 0);
            break;
        case 'o':
            filename = optarg;
            break;
        case 's':
            seed = (int)strtol(optarg, NULL, 0);
            break;
        case 'h':
            printf("Usage:\t%s [-n]\n", argv[0]);
            printf("\t-n\tnumber of particles\n");
            printf("\t-t\tnumber of iterations\n");
            printf("\t-s\tseed of random numbers\n");
            printf("\t-o\toutput filename\n");
            printf("\t-h\thelp\n");
            // MPI_Finalize();
            return (0);
        default:
            fprintf(stderr, "%s:\tinvalid option -- %c\n", argv[0], opt);
            fprintf(stderr, "Try `%s -h' for more information.\n", argv[0]);
            // MPI_Finalize();
            return (1);
        }
    }

    // for get_x_current
    srand48(seed);

    double* x = (double*)malloc(n * sizeof(double));
    double potential;
    double potential_delta;
    double x_current;
    // debug
    // int mutation_counter = 0;
    // int total_iteration;
    // double mutation_percent;
    // double mutation_ratio;

    init_particles(n, x);

    printf("Statistics:\n");
    printf("n\t%d\n", n);
    printf("#OpenMP\t%d\n", omp_get_max_threads());

    int iterations;
    // do {
    for (iterations = 0; iterations < t; iterations++) {
        // mutate x[i]. Do not mutate the one on the right boundary (x = L)
        for (int i = 0; i < n - 1; i++) {
            x_current = get_x_current(n, x, i, 100);
            // printf("%d\t%f\n", i, x_current); // debug
            potential_delta = get_potential_delta(n, x, i, x_current);
            if (potential_delta < 0) {
                x[i] = x_current;
                // debug
                // mutation_counter++;
                // if (i == n - 2){
                //     total_iteration = (n - 1) * (iterations + 1);
                //     // mutation_percent = mutation_counter / total_iteration;
                //     mutation_ratio = (double)total_iteration / mutation_counter;
                //     printf("%d / %d = %f\n", total_iteration, mutation_counter, mutation_ratio);
                // }
            }
            // debug
            // potential = get_potential(n, x) / n / n;
            // potential_delta = potential_delta / n / n;
            // printf("%d\t%f\t%f\t%f\n", i, potential, potential_delta, x_current);
        }
    }
    // TODO
    // } while (moveAgain(n - 1, particles, errorBound, iterations));

    // normalize potential
    potential = get_potential(n, x) / n / n;

    print_all(filename, n, x);
    printf("Potential is %f\n", potential);

    printf("\nStatistics:\n");
    printf("Iterations\t%d\n", iterations);

    free(x);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Compute the dot product of two vectors of length n
double dot_product(int n, double* a, double* b) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Build dense matrix A with entries A_ij = (N - |i - j|) / N
void build_dense_matrix(int N, double* A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (double)(N - abs(i - j)) / N;
        }
    }
}

// Initialize right-hand side vector b with all 1s
void build_rhs_vector(int N, double* b) {
    for (int i = 0; i < N; ++i) {
        b[i] = 1.0;
    }
}

// Conjugate Gradient method with residual recording
// Returns the number of iterations used until convergence
int cg_dense(int N, double* A, double* b, double* x, double* residuals, int maxiter, double reltol) {
    double *r = malloc(N * sizeof(double));
    double *p = malloc(N * sizeof(double));
    double *Ap = malloc(N * sizeof(double));

    // Initialize x with small random perturbations to avoid numerical pathologies
    for (int i = 0; i < N; ++i) {
        x[i] = 1e-6 * (2.0 * drand48() - 1.0); // random value in [-1e-6, 1e-6]
    }

    // Compute initial residual r = b - A * x
    for (int i = 0; i < N; ++i) {
        r[i] = b[i];
        for (int j = 0; j < N; ++j) {
            r[i] -= A[i * N + j] * x[j];
        }
    }

    // Initialize p = r
    for (int i = 0; i < N; ++i) p[i] = r[i];

    double r0_norm = sqrt(dot_product(N, r, r));
    double rsold = r0_norm * r0_norm;

    int k;
    for (k = 0; k < maxiter; ++k) {
        // Ap = A * p
        for (int i = 0; i < N; ++i) {
            Ap[i] = 0.0;
            for (int j = 0; j < N; ++j) {
                Ap[i] += A[i * N + j] * p[j];
            }
        }

        double alpha = rsold / dot_product(N, p, Ap);

        // Update x and residual r
        for (int i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double rsnew = dot_product(N, r, r);
        residuals[k] = sqrt(rsnew); // Save residual norm at iteration k

        // Check stopping condition
        if (residuals[k] <= reltol * r0_norm) break;

        // Update search direction
        double beta = rsnew / rsold;
        for (int i = 0; i < N; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    // Free dynamically allocated memory
    free(r);
    free(p);
    free(Ap);
    return k + 1; // Return number of iterations
}

int main() {
    srand48(time(NULL)); // Seed random number generator

    int Ns[] = {100, 1000, 10000,100000};  // Problem sizes to test
    int numTests = 4;
    int maxiter = 10000;
    double reltol = sqrt(DBL_EPSILON);  // Relative tolerance for convergence

    printf("%8s %12s %15s %20s\n", "N", "Iterations", "Time(s)", "Final Residual");
    printf("-------------------------------------------------------------------------------------------\n");

    for (int t = 0; t < numTests; ++t) {
        int N = Ns[t];

        double* A = malloc(N * N * sizeof(double));
        double* b = malloc(N * sizeof(double));
        double* x = malloc(N * sizeof(double));
        double* residuals = malloc(maxiter * sizeof(double));

        // Set up A and b
        build_dense_matrix(N, A);
        build_rhs_vector(N, b);

        // Run CG and measure time
        clock_t start = clock();
        int iterations = cg_dense(N, A, b, x, residuals, maxiter, reltol);
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

        // Output performance data
        printf("%8d %12d %15.4f %20.4e\n", N, iterations, time_spent, residuals[iterations - 1]);

        // Save residuals to file for later analysis
        char filename[64];
        sprintf(filename, "residuals_N%d.txt", N);
        FILE* fp = fopen(filename, "w");
        for (int i = 0; i < iterations; ++i) {
            fprintf(fp, "%e\n", residuals[i]);
        }
        fclose(fp);

        // Free memory
        free(A);
        free(b);
        free(x);
        free(residuals);
    }

    return 0;
}
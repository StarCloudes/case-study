// Implementation of Question 3.3: Convergence of CG for a dense matrix

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Dot product of two vectors
double dot_product(int n, double* a, double* b) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Build dense matrix A_ij = (N - |i - j|) / N
void build_dense_matrix(int N, double* A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (double)(N - abs(i - j)) / N;
        }
    }
}

// Fill b with all ones
void build_rhs_vector(int N, double* b) {
    for (int i = 0; i < N; ++i) {
        b[i] = 1.0;
    }
}

// Conjugate Gradient for dense matrix with residual tracking
int cg_dense(int N, double* A, double* b, double* x, double* residuals, int maxiter, double reltol) {
    double *r = malloc(N * sizeof(double));
    double *p = malloc(N * sizeof(double));
    double *Ap = malloc(N * sizeof(double));

    // Initialize x with a small random perturbation
    for (int i = 0; i < N; ++i) {
        x[i] = 1e-6 * (2.0 * drand48() - 1.0);
    }

    // r = b - A * x
    for (int i = 0; i < N; ++i) {
        r[i] = b[i];
        for (int j = 0; j < N; ++j) {
            r[i] -= A[i * N + j] * x[j];
        }
    }

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
        for (int i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double rsnew = dot_product(N, r, r);
        residuals[k] = sqrt(rsnew);

        if (residuals[k] <= reltol * r0_norm) break;

        double beta = rsnew / rsold;
        for (int i = 0; i < N; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    free(r);
    free(p);
    free(Ap);
    return k + 1;
}

int main() {
    srand48(time(NULL)); // seed random generator for perturbation

    int Ns[] = {100, 1000, 10000};
    int numTests = 3;
    int maxiter = 10000;
    double reltol = sqrt(DBL_EPSILON);

    printf("%8s %12s %15s %20s %15s\n", "N", "Iterations", "Time(s)", "Final Residual", "    Estimated kappa");
    printf("-------------------------------------------------------------------------------------------\n");

    for (int t = 0; t < numTests; ++t) {
        int N = Ns[t];

        double* A = malloc(N * N * sizeof(double));
        double* b = malloc(N * sizeof(double));
        double* x = malloc(N * sizeof(double));
        double* residuals = malloc(maxiter * sizeof(double));

        build_dense_matrix(N, A);
        build_rhs_vector(N, b);

        clock_t start = clock();
        int iterations = cg_dense(N, A, b, x, residuals, maxiter, reltol);
        clock_t end = clock();

        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        double kappa = 1 + 2.0 * (N - 1);

        printf("%8d %12d %15.4f %20.4e %15.0f\n", N, iterations, time_spent, residuals[iterations - 1], kappa);

        // Save residuals for Python plotting
        char filename[64];
        sprintf(filename, "residuals_N%d.txt", N);
        FILE* fp = fopen(filename, "w");
        for (int i = 0; i < iterations; ++i) {
            fprintf(fp, "%e\n", residuals[i]);
        }
        fclose(fp);

        free(A);
        free(b);
        free(x);
        free(residuals);
    }
    return 0;
}

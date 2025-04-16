#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//  Compute f(x, y) = 2π² sin(πx) sin(πy)
double f_function(double x, double y) {
    return 2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

// Compute the dot product of two vectors of length n
double dot_product(int n, double* a, double* b) {
    double dot = 0.0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

// Compute the matrix-vector product A*x and store result in Ax
// A comes from a 5-point stencil: for interior node (i,j),
// A*x = 4*x[i,j] - (x[i-1,j] + x[i+1,j] + x[i,j-1] + x[i,j+1])
// The grid has (N-1) x (N-1) interior nodes, stored in row-major order
void apply_A(int N, double* x, double* Ax) {
    int gridSize = N - 1;  // Number of interior nodes per row
    for (int j = 0; j < gridSize; j++) {
        for (int i = 0; i < gridSize; i++) {
            int idx = i + j * gridSize;
            double sum = 4.0 * x[idx];
            if (i > 0)
                sum -= x[idx - 1];    // Left neighbor
            if (i < gridSize - 1)
                sum -= x[idx + 1];  // Right neighbor
            if (j > 0)
                sum -= x[idx - gridSize]; // Bottom neighbor
            if (j < gridSize - 1)
                sum -= x[idx + gridSize];  // Top neighbor
            Ax[idx] = sum;
        }
    }
}

// Conjugate Gradient solver for A*x = b
// N: number of intervals in [0,1], so interior points are (N-1)x(N-1)
// b: right-hand side vector (length (N-1)^2)
// x: solution vector (initialized with random perturbation)
// iterations: returns actual number of iterations performed
// tol: tolerance; maxiter: maximum number of iterations
void cg_solver(int N, double* b, double* x, int* iterations, double tol, int maxiter) {
    int gridSize = N - 1;
    int n = gridSize * gridSize;
    
    double *r = (double*)malloc(n * sizeof(double));
    double *p = (double*)malloc(n * sizeof(double));
    double *Ap = (double*)malloc(n * sizeof(double));

    // Initialize x with a small random perturbation (avoid starting exactly at zero)
    for (int i = 0; i < n; i++) {
        x[i] = 1e-6 * (2.0 * drand48() - 1.0);
    }
    
    // Compute initial residual r = b - A*x
    apply_A(N, x, Ap);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }
    
    double rsold = dot_product(n, r, r);
    *iterations = 0;
    
    for (int k = 0; k < maxiter; k++) {
        apply_A(N, p, Ap);
        double pAp = dot_product(n, p, Ap);
        if (fabs(pAp) < 1e-16) {
            printf("pAp is nearly zero, stopping iteration.\n");
            *iterations = k;
            break;
        }
        double alpha = rsold / pAp;
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        double rsnew = dot_product(n, r, r);
        if (sqrt(rsnew) < tol) {
            *iterations = k + 1;
            break;
        }
        double beta = rsnew / rsold;
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
        *iterations = k + 1;
    }
    
    free(r);
    free(p);
    free(Ap);
}

int main() {
    srand48(time(NULL)); // Seed the drand48 random number generator

    // Set N values to test, e.g., N = 8, 16, 32, 64, 128, 256
    int numTests = 6;
    int N_values[] = {8, 16, 32, 64, 128, 256};
    double tol = 1e-8;
    int maxiter = 10000;
    
    // Print N, Iterations, Time(s)
    printf("%8s %12s %15s\n", "N", "Iterations", "Time(s)");
    printf("-------------------------------------------------\n");
    
    for (int t = 0; t < numTests; t++) {
        int N = N_values[t];
        int gridSize = N - 1;
        int n = gridSize * gridSize;
        double h = 1.0 / N;
        
        double* b = (double*)malloc(n * sizeof(double));
        double* x = (double*)malloc(n * sizeof(double));
        
        // Construct right-hand side vector b
        for (int j = 0; j < gridSize; j++) {
            for (int i = 0; i < gridSize; i++) {
                int idx = i + j * gridSize;
                double x_coord = (i + 1) * h;
                double y_coord = (j + 1) * h;
                b[idx] = h * h * f_function(x_coord, y_coord);
            }
        }
        
        int iters = 0;
        clock_t start = clock();
        cg_solver(N, b, x, &iters, tol, maxiter);
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        
        printf("%8d %12d %15.4f\n", N, iters, time_spent);
        
        // Output solution to CSV for plotting when N = 256
        if (N == 256) {
            char sol_filename[64];
            sprintf(sol_filename, "u_solution_N%d.csv", N);
            FILE *fp_sol = fopen(sol_filename, "w");
            if (fp_sol) {
                // Output CSV file with x, y, u
                fprintf(fp_sol, "x,y,u\n");
                for (int j = 0; j < gridSize; j++) {
                    for (int i = 0; i < gridSize; i++) {
                        int idx = i + j * gridSize;
                        double x_coord = (i + 1) * h;
                        double y_coord = (j + 1) * h;
                        fprintf(fp_sol, "%.6e,%.6e,%.6e\n", x_coord, y_coord, x[idx]);
                    }
                }
                fclose(fp_sol);
            }
        }
        
        free(b);
        free(x);
    }
    
    return 0;
}
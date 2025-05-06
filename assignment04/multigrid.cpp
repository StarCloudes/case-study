#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <functional>

// Stores data for each multigrid level
struct GridLevel {
    int N;                      // Number of interior points per dimension
    std::vector<double> x;      // Solution vector
    std::vector<double> b;      // Right-hand side
    std::vector<double> r;      // Residual vector
};

// Assembles the right-hand side b for the 2D Poisson problem using a given function f(x, y)
void assemblePoissonRHS(int N,
                        const std::function<double(double,double)>& f,
                        std::vector<double>& b)
{
    double h = 1.0 / (N + 1);
    b.assign(N * N, 0.0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            int idx = j * N + i;
            double x = (i + 1) * h, y = (j + 1) * h;
            b[idx] = h * h * f(x, y);  // apply source function and scale
        }
}

// Applies weighted Jacobi smoothing to the current solution x
void smoothWeightedJacobi(std::vector<double>& x,
                          const std::vector<double>& b,
                          int N,
                          double omega,
                          int nu)
{
    int n = N * N;
    std::vector<double> x_new(n, 0.0);
    for (int it = 0; it < nu; ++it) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                int idx = j * N + i;
                double sum = 0.0;
                if (i > 0)       sum += x[idx - 1];
                if (i < N - 1)   sum += x[idx + 1];
                if (j > 0)       sum += x[idx - N];
                if (j < N - 1)   sum += x[idx + N];
                double Ax_i = 4.0 * x[idx] - sum;
                double r = b[idx] - Ax_i;
                x_new[idx] = x[idx] + omega * r / 4.0;
            }
        std::swap(x, x_new);
    }
}

// Computes the residual r = b - Ax using a 5-point stencil
void computeResidual(const std::vector<double>& x,
                     const std::vector<double>& b,
                     std::vector<double>& r,
                     int N)
{
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            int idx = j * N + i;
            double Ax = 4.0 * x[idx];
            if (i > 0)     Ax -= x[idx - 1];
            if (i < N - 1) Ax -= x[idx + 1];
            if (j > 0)     Ax -= x[idx - N];
            if (j < N - 1) Ax -= x[idx + N];
            r[idx] = b[idx] - Ax;
        }
}

// Restricts the fine grid residual to a coarse grid (simple averaging)
void restrictResidual(const std::vector<double>& r_f, int N_f, std::vector<double>& r_c)
{
    int N_c = N_f / 2;
    for (int j = 0; j < N_c; ++j)
        for (int i = 0; i < N_c; ++i) {
            int idx_c = j * N_c + i;
            int ii = 2 * i, jj = 2 * j;
            double sum = r_f[jj * N_f + ii];
            if (ii > 0) sum += r_f[jj * N_f + ii - 1];
            if (ii < N_f - 1) sum += r_f[jj * N_f + ii + 1];
            if (jj > 0) sum += r_f[(jj - 1) * N_f + ii];
            if (jj < N_f - 1) sum += r_f[(jj + 1) * N_f + ii];
            r_c[idx_c] = sum / 5.0;
        }
}

// Prolongates the coarse grid correction back to the fine grid (simple interpolation)
void prolongateError(const std::vector<double>& e_c, int N_c, std::vector<double>& e_f)
{
    int N_f = 2 * N_c;
    std::fill(e_f.begin(), e_f.end(), 0.0);
    for (int j = 0; j < N_c; ++j)
        for (int i = 0; i < N_c; ++i) {
            int idx_c = j * N_c + i;
            int ii = 2 * i, jj = 2 * j;
            e_f[jj * N_f + ii] += e_c[idx_c];
            if (ii + 1 < N_f) e_f[jj * N_f + ii + 1] += 0.5 * e_c[idx_c];
            if (jj + 1 < N_f) e_f[(jj + 1) * N_f + ii] += 0.5 * e_c[idx_c];
            if (ii + 1 < N_f && jj + 1 < N_f)
                e_f[(jj + 1) * N_f + ii + 1] += 0.25 * e_c[idx_c];
        }
}

// Approximates the coarse grid solve by doing extra Jacobi smoothing (instead of exact solve)
void directSolve(std::vector<double>& x, const std::vector<double>& b, int N)
{
    x.assign(N * N, 0.0);
    smoothWeightedJacobi(x, b, N, 0.8, 30);
}

// Recursive V-cycle implementation
void Vcycle(std::vector<GridLevel>& levels, int l, double omega, int nu)
{
    auto& level = levels[l];
    int N = level.N;

    smoothWeightedJacobi(level.x, level.b, N, omega, nu);
    computeResidual(level.x, level.b, level.r, N);

    if (l + 1 == (int)levels.size()) {
        directSolve(level.x, level.b, N);
        return;
    }

    auto& next = levels[l + 1];
    restrictResidual(level.r, N, next.b);
    std::fill(next.x.begin(), next.x.end(), 0.0);
    Vcycle(levels, l + 1, omega, nu);

    std::vector<double> e_f(N * N, 0.0);
    prolongateError(next.x, next.N, e_f);
    for (int i = 0; i < N * N; ++i)
        level.x[i] += e_f[i];

    smoothWeightedJacobi(level.x, level.b, N, omega, nu);
}

int main()
{
    int N0 = 32;               // Finest grid size (NxN)
    int levels_count = 4;      // Number of multigrid levels
    double omega = 0.67;       // Jacobi relaxation parameter
    int nu = 10;               // Smoothing steps
    double tol = 1e-7;         // Convergence tolerance

    // Initialize multigrid levels
    std::vector<GridLevel> levels(levels_count);
    for (int l = 0; l < levels_count; ++l) {
        int N = N0 >> l;
        int n = N * N;
        levels[l].N = N;
        levels[l].x.assign(n, 0.0);
        levels[l].b.assign(n, 0.0);
        levels[l].r.assign(n, 0.0);

        if (l == 0) {
            auto f = [](double x, double y) {
                const double pi = 3.141592653589793;
                return 2 * pi * pi * sin(pi * x) * sin(pi * y);
            };
            assemblePoissonRHS(N, f, levels[l].b);
        }
    }

    // Run V-cycles until the solution converges
    for (int iter = 0; iter < 500; ++iter) {
        Vcycle(levels, 0, omega, nu);
        computeResidual(levels[0].x, levels[0].b, levels[0].r, levels[0].N);
        double max_r = 0;
        for (double v : levels[0].r)
            max_r = std::max(max_r, std::abs(v));

        std::cout << "Iter " << std::setw(3) << iter
                  << ": max|r| = " << max_r << "\n";

        if (max_r < tol) {
            std::cout << "Converged after " << iter + 1 << " V-cycles.\n";
            break;
        }
    }

    return 0;
}
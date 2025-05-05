#include <vector>
#include <functional>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <utility> 

/**
 * @brief Assemble the linear system A*y = b for the 2D Poisson problem
 *        using a 5-point finite difference stencil.
 * @param N        Number of interior points in each dimension (number of unknowns = N×N).
 * @param f        Right-hand side function f(x,y).
 * @param A        Output dense matrix of size (N*N)×(N*N).
 * @param b        Output right-hand side vector of length N*N.
 */
void assemblePoisson2D(int N,
                       const std::function<double(double,double)>& f,
                       std::vector<std::vector<double>>& A,
                       std::vector<double>& b)
{
    double h = 1.0 / (N + 1);  // grid spacing h
    int n = N * N;             // total number of unknowns
    A.assign(n, std::vector<double>(n, 0.0));
    b.assign(n, 0.0);

    // loop over each interior grid point (i,j)
    for (int j = 1; j <= N; ++j) {
        for (int i = 1; i <= N; ++i) {
            int k = (j-1)*N + (i-1);  // map (i,j) to index k

            // central coefficient (diagonal)
            A[k][k] = 4.0;
            // neighbor coefficients: left, right, down, up
            if (i > 1)   A[k][k-1]   = -1.0;
            if (i < N)   A[k][k+1]   = -1.0;
            if (j > 1)   A[k][k-N]   = -1.0;
            if (j < N)   A[k][k+N]   = -1.0;

            // right-hand side: b_k = h^2 * f(x_i, y_j)
            double x = i * h;
            double y = j * h;
            b[k] = h*h * f(x, y);
        }
    }
}

/// @brief  Grid level data
struct GridLevel {
    int N;                                     // number of interior points per dimension
    std::vector<std::vector<double>> A;        // system matrix (dense)
    std::vector<double> b;                     // right-hand side
    std::vector<double> x;                     // solution
    std::vector<double> r;                     // residual
};

/**
 * @brief Weighted Jacobi
 *        Performs ν iterations of weighted Jacobi: x ← x + ω D⁻¹(b − A x).
 * @param A       system matrix
 * @param x       current solution
 * @param b       right-hand side
 * @param omega   relaxation parameter
 * @param nu      number of iterations
 */
void smoothWeightedJacobi(const std::vector<std::vector<double>>& A,
                          std::vector<double>& x,
                          const std::vector<double>& b,
                          double omega,
                          int nu)
{
    int n = x.size();
    std::vector<double> x_new(n);
    //  ν iterations
    for (int it = 0; it < nu; ++it) {
        for (int i = 0; i < n; ++i) {
            double sigma = 0;
            for (int j = 0; j < n; ++j) {
                if (j != i) sigma += A[i][j] * x[j];
            }
            // x_new[i] = x[i] + ω*(b[i] − Σ A[i][j] x[j]) / A[i][i]
            x_new[i] = x[i] + omega * (b[i] - sigma - A[i][i] * x[i]) / A[i][i];
        }
        x.swap(x_new);
    }
}



/**
 * @brief Solve A x = b by naive Gaussian elimination with partial pivoting.
 *        Only appropriate for small systems (coarse grid).
 * @param A   Coefficient matrix (will be modified).
 * @param x   Solution vector (output).
 * @param b   Right-hand side vector.
 */
void directSolve(std::vector<std::vector<double>>& A,
                 std::vector<double>& x,
                 const std::vector<double>& b) {
    int n = A.size();
    // build augmented matrix
    std::vector<std::vector<double>> M(n, std::vector<double>(n+1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) M[i][j] = A[i][j];
        M[i][n] = b[i];
    }
    // forward elimination
    for (int k = 0; k < n; ++k) {
        // pivot
        int piv = k;
        for (int i = k+1; i < n; ++i)
            if (std::fabs(M[i][k]) > std::fabs(M[piv][k])) piv = i;
        std::swap(M[k], M[piv]);
        // eliminate
        for (int i = k+1; i < n; ++i) {
            double factor = M[i][k] / M[k][k];
            for (int j = k; j <= n; ++j)
                M[i][j] -= factor * M[k][j];
        }
    }
    // back substitution
    x.assign(n, 0.0);
    for (int i = n-1; i >= 0; --i) {
        double sum = M[i][n];
        for (int j = i+1; j < n; ++j)
            sum -= M[i][j] * x[j];
        x[i] = sum / M[i][i];
    }
}

/**
 * @brief r = b − A x
 * @param A   system matrix
 * @param x   current solution
 * @param b   right-hand side
 * @param r   output residual
 */
void computeResidual(const std::vector<std::vector<double>>& A,
                     const std::vector<double>& x,
                     const std::vector<double>& b,
                     std::vector<double>& r)
{
    int n = x.size();
    for (int i = 0; i < n; ++i) {
        double Ax_i = 0;
        for (int j = 0; j < n; ++j) {
            Ax_i += A[i][j] * x[j];
        }
        r[i] = b[i] - Ax_i;
    }
}

/**
 * @brief Restriction: restrict fine-grid residual r_f to coarse-grid residual r_c.
 *        Uses simple injection or averaging; here we average over each 2×2 block.
 * @param r_f   N_f = N_f^2 / fine residual
 * @param N_f   N_f / fine grid interior size
 * @param r_c   N_c = N_c^2 / coarse residual (output)
 */
void restrictResidual(const std::vector<double>& r_f,
                      int N_f,
                      std::vector<double>& r_c)
{
    int N_c = N_f / 2;  // coarse interior points
    for (int j = 1; j <= N_c; ++j) {
        for (int i = 1; i <= N_c; ++i) {
            int idx_c = (j-1)*N_c + (i-1);
            // corresponding fine-grid center point: (2i, 2j)
            int ii = 2*i, jj = 2*j;
            // average the residuals of the four surrounding fine points
            double sum = 0;
            sum += r_f[(jj-2)*(N_f) + (ii-2)];
            sum += r_f[(jj-2)*(N_f) + (ii   -1)];
            sum += r_f[(jj   -1)*(N_f) + (ii-2)];
            sum += r_f[(jj   -1)*(N_f) + (ii   -1)];
            r_c[idx_c] = sum * 0.25;
        }
    }
}

/**
 * @brief Prolongation: interpolate coarse-grid error e_c to fine-grid error e_f.
 *        Performs bilinear interpolation.
 * @param e_c   (n_c = N_c^2) / coarse error
 * @param N_c   N_c / coarse interior size
 * @param e_f   (n_f = N_f^2) / fine error (output)
 */
void prolongateError(const std::vector<double>& e_c,
                     int N_c,
                     std::vector<double>& e_f)
{
    int N_f = 2 * N_c;  // fine interior
    // initialize to 0
    std::fill(e_f.begin(), e_f.end(), 0.0);
    // interpolation: assign values at matching nodes, then average for neighbors
    for (int j = 1; j <= N_c; ++j) {
        for (int i = 1; i <= N_c; ++i) {
            int idx_c = (j-1)*N_c + (i-1);
            int ii = 2*i-1, jj = 2*j-1;  // 0-based
            e_f[(jj-1)*N_f + (ii-1)] += e_c[idx_c];
            // horizontal direction
            if (ii < N_f) e_f[(jj-1)*N_f + ii]     += 0.5 * e_c[idx_c];
            // vertical direction
            if (jj < N_f) e_f[(jj)*N_f   + (ii-1)] += 0.5 * e_c[idx_c];
            // diagonal
            if (ii < N_f && jj < N_f) e_f[jj*N_f + ii] += 0.25 * e_c[idx_c];
        }
    }
}

/**
 * @brief Recursive V-cycle implementation
 * @param levels    vector of GridLevel
 * @param l         current level
 * @param omega     relaxation parameter
 * @param nu        smoothing steps
 */
void Vcycle(std::vector<GridLevel>& levels,
            int l,
            double omega,
            int nu)
{
    auto& level = levels[l];
    int n = level.x.size();

    // 1) pre-smoothing
    smoothWeightedJacobi(level.A, level.x, level.b, omega, nu);

    // 2) compute residual r_l = b_l − A_l x_lv
    computeResidual(level.A, level.x, level.b, level.r);

    // if at the coarsest level, solve directly with Gaussian elimination
    if (l + 1 == static_cast<int>(levels.size())) {
        directSolve(level.A, level.x, level.b);
        return;
    }

    // 3) restrict residual to the next level b_{l+1}
    auto& next = levels[l+1];
    restrictResidual(level.r, level.N, next.b);

    // 4) initialize solution to zero at the next level
    std::fill(next.x.begin(), next.x.end(), 0.0);

    // 5) recursively call V-cycle
    Vcycle(levels, l+1, omega, nu);

    // 6) prolongation correction: x_l += P e_{l+1}
    std::vector<double> e_f(n, 0.0);
    prolongateError(next.x, next.N, e_f);
    for (int i = 0; i < n; ++i) level.x[i] += e_f[i];

    // 7) post-smoothing
    smoothWeightedJacobi(level.A, level.x, level.b, omega, nu);
}


int main()
{
    int N0 = 32;          // reduced from 128 to avoid excessive memory
    int levels_count = 4; // how many levels
    double omega = 0.67;  // 0<ω<1 / relaxation
    int nu = 10;          // smoothing steps

    if (!(omega>0 && omega<2)) {
        std::cerr<<"Error: omega must be in (0,2).\n"; return -1;
    }
    if (nu <= 0) {
        std::cerr<<"Error: nu must be positive.\n"; return -1;
    }
    if (levels_count < 2) {
        std::cerr<<"Error: at least two levels required.\n"; return -1;
    }

    // construct multigrid levels
    std::vector<GridLevel> levels(levels_count);
    for (int l = 0; l < levels_count; ++l) {
        int N = N0 >> l;        // coarsening: halve the grid points at each level
        levels[l].N = N;
        int n = N * N;
        levels[l].A.assign(n, std::vector<double>(n, 0.0));
        levels[l].b.assign(n, 0.0);
        levels[l].x.assign(n, 0.0);
        levels[l].r.assign(n, 0.0);

        // assemble A and b for the finest level; for other levels, initialize b to 0 and reuse or reassemble A
        if (l == 0) {
            auto f = [](double x, double y) {
                const double pi = 3.141592653589793;
                return 2.0*pi*pi * sin(pi*x) * sin(pi*y);
            };
            assemblePoisson2D(N, f, levels[l].A, levels[l].b);
        } else {
            assemblePoisson2D(levels[l].N, 
                              [&](double x,double y){ return 0.0; },
                              levels[l].A, levels[l].b);
        }
    }

    // loop V-cycles until the residual is below the tolerance
    double tol = 1e-7;
    for (int iter = 0; iter < 500; ++iter) {
        Vcycle(levels, 0, omega, nu);
        computeResidual(levels[0].A, levels[0].x, levels[0].b, levels[0].r);
        double norm_r = 0;
        for (double v : levels[0].r) norm_r = std::max(norm_r, std::fabs(v));
        std::cout << "Iter " << std::setw(2) << iter
                  << "  max|r| = " << norm_r << "\n";
        if (norm_r < tol) {
            std::cout << "Converged after " << iter+1 << " V-cycles.\n";
            break;
        }
    }

    return 0;
}
#include <vector>
#include <functional>
#include <iostream>
#include <cmath>
#include <algorithm>

/**
 * @brief Assemble the linear system A*y = b for the 2D Poisson problem
 *        using a 5-point finite difference stencil.
 *
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
    std::cout << "N=" << N
              << ", h=" << h
              << ", n=" << n << std::endl;

    // Resize output containers
    A.assign(n, std::vector<double>(n, 0.0));
    b.assign(n, 0.0);

    // loop over each interior grid point (i,j)
    for (int j = 1; j <= N; ++j) {
        for (int i = 1; i <= N; ++i) {
            int k = (j-1)*N + (i-1);  // map (i,j) to index k

            // central coefficient (diagonal)
            A[k][k] =  4.0;

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

// ---------   simple test --------------
int main() {
    int N = 4;  // small-scale test
    std::cout << "Starting assembly with N=" << N << std::endl;
    std::vector<std::vector<double>> A;
    std::vector<double> b;

    // RHS function f(x,y) = 2*pi^2 sin(pi x) sin(pi y)
    std::function<double(double,double)> f = [](double x, double y) {
        const double pi = 3.141592653589793;
        return 2.0*pi*pi * sin(pi*x) * sin(pi*y);
    };

    assemblePoisson2D(N, f, A, b);
    std::cout << "Assembly done. A size: "
              << A.size() << " x " << (A.empty() ? 0 : A[0].size())
              << ", b size: " << b.size() << std::endl;
              
    // Print first few entries of first row of A
    std::cout << "A[0][0..3]= ";
    for (int j = 0; j < std::min((size_t)4, A[0].size()); ++j)
        std::cout << A[0][j] << " ";
    std::cout << std::endl;

    // print b vector
    std::cout << "b = [ ";
    for (double v : b) std::cout << v << " ";
    std::cout << "]\n";

    return 0;
}
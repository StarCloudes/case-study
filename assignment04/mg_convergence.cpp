#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <functional>

using Vec = std::vector<double>;
using Clock = std::chrono::high_resolution_clock;

int g_coarse_solves = 0;

// Matrix-free weighted Jacobi smoother
void smoothWeightedJacobiMF(int N, Vec& x, const Vec& b,
                            double omega, int nu) {
    int n = N * N;
    double h2 = 1.0 / (N + 1) / (N + 1);
    Vec tmp(n);
    for (int it = 0; it < nu; ++it) {
        for (int k = 0; k < n; ++k) {
            int i = k % N, j = k / N;
            double Ax = 4.0 * x[k];
            if (i > 0)   Ax -= x[k - 1];
            if (i < N - 1) Ax -= x[k + 1];
            if (j > 0)   Ax -= x[k - N];
            if (j < N - 1) Ax -= x[k + N];
            double r = b[k] - Ax * h2;
            tmp[k] = x[k] + omega * (h2 / 4.0) * r;
        }
        x.swap(tmp);
    }
}

// Matrix-free residual: r = b - A*x
void computeResidualMF(int N, const Vec& x, const Vec& b, Vec& r) {
    int n = N * N;
    double h2 = 1.0 / (N + 1) / (N + 1);
    r.assign(n, 0.0);
    for (int k = 0; k < n; ++k) {
        int i = k % N, j = k / N;
        double Ax = 4.0 * x[k];
        if (i > 0)   Ax -= x[k - 1];
        if (i < N - 1) Ax -= x[k + 1];
        if (j > 0)   Ax -= x[k - N];
        if (j < N - 1) Ax -= x[k + N];
        r[k] = b[k] - Ax * h2;
    }
}

// Restriction (average 2x2 block)
void restrictResidual(const Vec& rf, int Nf, Vec& rc) {
    int Nc = Nf / 2;
    rc.assign(Nc * Nc, 0.0);
    for (int j = 0; j < Nc; ++j) {
        for (int i = 0; i < Nc; ++i) {
            int kc = j * Nc + i;
            int kf = (2 * j) * Nf + 2 * i;
            rc[kc] = 0.25 * (rf[kf] + rf[kf + 1] + rf[kf + Nf] + rf[kf + Nf + 1]);
        }
    }
}

// Prolongation (bilinear interpolation)
void prolongateError(const Vec& ec, int Nc, Vec& ef) {
    int Nf = Nc * 2;
    ef.assign(Nf * Nf, 0.0);
    for (int j = 0; j < Nc; ++j) {
        for (int i = 0; i < Nc; ++i) {
            int kc = j * Nc + i;
            int kf = (2 * j) * Nf + 2 * i;
            double v = ec[kc];
            ef[kf] += v;
            ef[kf + 1] += 0.5 * v;
            ef[kf + Nf] += 0.5 * v;
            ef[kf + Nf + 1] += 0.25 * v;
        }
    }
}

// Iterative solver for coarse level
void solveCoarseIter(int N, Vec& x, const Vec& b,
                     double omega, int nu,
                     double tol, int maxit) {
    ++g_coarse_solves;
    x.assign(N * N, 0.0);
    Vec r;
    for (int iter = 0; iter < maxit; ++iter) {
        smoothWeightedJacobiMF(N, x, b, omega, nu);
        computeResidualMF(N, x, b, r);
        double rmax = 0;
        for (double v : r) rmax = std::max(rmax, std::fabs(v));
        if (rmax < tol) break;
    }
}

// Recursive V-cycle
void Vcycle(const std::vector<int>& Ns,
            std::vector<Vec>& bs,
            std::vector<Vec>& xs,
            std::vector<Vec>& rs,
            int lvl,
            double omega, int nu,
            double tol_c, int maxit_c) {
    int N = Ns[lvl];
    Vec& b = bs[lvl], &x = xs[lvl], &r = rs[lvl];

    smoothWeightedJacobiMF(N, x, b, omega, nu);
    computeResidualMF(N, x, b, r);

    if (lvl + 1 == (int)Ns.size()) {
        solveCoarseIter(N, x, r, omega, nu, tol_c, maxit_c);
        return;
    }

    restrictResidual(r, N, bs[lvl + 1]);
    std::fill(xs[lvl + 1].begin(), xs[lvl + 1].end(), 0.0);
    Vcycle(Ns, bs, xs, rs, lvl + 1, omega, nu, tol_c, maxit_c);
    Vec ef;
    prolongateError(xs[lvl + 1], Ns[lvl + 1], ef);
    for (size_t k = 0; k < x.size(); ++k) x[k] += ef[k];
    smoothWeightedJacobiMF(N, x, b, omega, nu);
}

int main() {
    using namespace std::chrono;
    std::vector<int> Ns0 = {16, 32, 64, 128, 256};
    std::vector<std::pair<int, bool>> configs = {
        {2, false},  // 2-level MG
        {5, true}    // full MG to Nc=8
    };

    double omega = 0.67;
    int nu = 5;
    double tol = 1e-7;
    int maxit = 500;
    double tol_c = 1e-7;
    int maxit_c = 500;

    std::ofstream fout("mg_results.csv");
    fout << "N,levels,iters,time_s,coarse_solves\n";

    for (int N0 : Ns0) {
        for (auto cfg : configs) {
            int L = cfg.first;
            bool fix_coarse = cfg.second;
            std::vector<double> res_hist;

            std::vector<int> Ns(L);
            std::vector<Vec> bs(L), xs(L), rs(L);
            for (int l = 0; l < L; ++l) {
                int N = N0 >> l;
                if (fix_coarse && N < 8) N = 8;
                Ns[l] = N;
                int n = N * N;
                bs[l].assign(n, 0.0);
                xs[l].assign(n, 0.0);
                rs[l].assign(n, 0.0);
                if (l == 0) {
                    double h = 1.0 / (N + 1);
                    for (int k = 0; k < n; ++k) {
                        int i = k % N, j = k / N;
                        double x = (i + 1) * h, y = (j + 1) * h;
                        bs[l][k] = h * h * 2 * M_PI * M_PI *
                                   std::sin(M_PI * x) * std::sin(M_PI * y);
                    }
                }
            }

            g_coarse_solves = 0;
            auto t0 = Clock::now();
            int it;
            for (it = 0; it < maxit; ++it) {
                Vcycle(Ns, bs, xs, rs, 0, omega, nu, tol_c, maxit_c);
                computeResidualMF(Ns[0], xs[0], bs[0], rs[0]);
                double rmax = 0;
                for (double v : rs[0]) rmax = std::max(rmax, std::fabs(v));
                res_hist.push_back(rmax);
                if (rmax < tol) break;
            }
            auto t1 = Clock::now();
            double elapsed = duration<double>(t1 - t0).count();

            fout << N0 << "," << L << "," << it + 1 << "," << elapsed << "," << g_coarse_solves << "\n";
            std::cout << "N=" << N0
                      << " L=" << L
                      << " iters=" << it + 1
                      << " time=" << std::fixed << std::setprecision(3) << elapsed << "s"
                      << " solves=" << g_coarse_solves << "\n";

            // Save residual history
            std::string fname = "residuals_N" + std::to_string(N0) + "_L" + std::to_string(L) + ".csv";
            std::ofstream rfile(fname);
            for (double r : res_hist) rfile << r << "\n";
        }
    }
    return 0;
}
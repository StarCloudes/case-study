import numpy as np
import matplotlib.pyplot as plt

def compute_kappa(N):
    print(f"\nComputing true κ for N={N} ...")
    A = np.fromfunction(lambda i, j: (N - np.abs(i - j)) / N, (N, N))
    eigs = np.linalg.eigvalsh(A) 
    kappa = eigs[-1] / eigs[0]
    print(f"λ_max = {eigs[-1]:.3e}, λ_min = {eigs[0]:.3e}, κ = {kappa:.3e}")
    return kappa

def plot_residual_vs_bound(N):
    # Load residual data
    residuals = np.loadtxt(f"residuals_N{N}.txt")
    k = np.arange(len(residuals))
    r0 = residuals[0]

    # Get condition number
    kappa = compute_kappa(N)
    rho = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    bound = 2 * (rho ** k) * r0

    # Plot
    plt.figure(figsize=(8, 5))
    plt.semilogy(k, residuals, label='Actual Residual $\\|r_k\\|$')
    plt.semilogy(k, bound, '--', label='Theoretical Bound $\\|e_k\\|$ (true $\\kappa$)')
    plt.xlabel('Iteration $k$')
    plt.ylabel('Residual Norm (log scale)')
    plt.title(f'CG Convergence with True $\\kappa$ (N={N})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"convergence_kappa_N{N}.png", dpi=150)
    plt.show()

# Draw plots for different N values
plot_residual_vs_bound(100)    
plot_residual_vs_bound(1000)   
plot_residual_vs_bound(10000)
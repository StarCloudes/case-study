# TSQR (Tall-Skinny QR) Decomposition with MPI and MATLAB

This repository contains a distributed TSQR (Tall-Skinny QR) implementation using MPI and OpenBLAS/LAPACK in C, along with a MATLAB version of the code. It includes:

- **MATLAB Version:**
  - **assignment01.mlx** (the main MATLAB livescript implementing TSQR)

- **C/MPI Version:**
  - **main.c** (the primary TSQR driver)
  - **matrix_utils.c/.h** (utility functions for matrix generation, printing, multiplication, etc.)
  - **tsqr_module.c/.h** (the modularized TSQR functions)
  - **performance_test.c** (a program to measure how TSQR scales for various matrix sizes)
  - **Makefile** (for building and running the C/MPI code)


## Prerequisites

### For the C/MPI Version

1. **MPI:**  
   Ensure you have an MPI implementation installed (e.g., MPICH, Open MPI). On macOS, you can install Open MPI via Homebrew:
   ```bash
   brew install open-mpi
   ```

2.	**OpenBLAS / LAPACK**:
    You need a BLAS/LAPACK library. On macOS, install them via Homebrew:
    ```bash
    brew install openblas lapack
    ```


## Building and Running

1. Run the TSQR program using 4 MPI processes:
    ```bash
    make run
    ```

2. Run the TSQR Performance Test using 4 MPI processes:
    ```bash
    make run_perf
    ```


# TSQR (Tall-Skinny QR) Decomposition with MPI and MATLAB

This repository contains a distributed TSQR (Tall-Skinny QR) implementation using `MPI` and `OpenBLAS/LAPACK` in C, along with a MATLAB version of the code. It includes:

- **MATLAB Version:**
  - **assignment01.mlx** (the main MATLAB livescript implementing TSQR)

- **C/MPI Version:**
  - **main.c** (the primary TSQR running program)
  - **scaling_test.c** (a program to measure how TSQR scales for various matrix sizes)
  - **matrix_utils.c/.h** (utility functions for matrix generation, printing, multiplication, etc.)
  - **tsqr_module.c/.h** (the modularized TSQR functions)
  - **Makefile** (for building and running the `C/MPI` code)

# Key Features of This Implementation
- **Distributed Computing:** Uses `MPI` to parallelize the QR factorization across multiple processes.
- **Step-by-Step Factorization:**
  1. **Local QR Factorization** – Each process computes QR for its submatrix.
  2. **Merging R Matrices** – The upper triangular R matrices are gathered and stacked.
  3. **Global QR on Merged R** – A final QR is performed on the merged R blocks.
  4. **Final Q Computation** – Each local Q is adjusted to form the global Q.
- **Validation:**
  - Computes Residual Error Analysis (`||QR - A||_F`) to check decomposition accuracy.
  - Verifies `Q^T * Q ≈ I` to ensure orthogonality.


# For the C/MPI Version

## I、Prerequisites

1. **MPI**  
    Ensure you have an `MPI` implementation installed. On macOS, you can install `Open MPI` via `Homebrew`:
    ```bash
    brew install open-mpi
    ```

2. **OpenBLAS / LAPACK**:
    You need a `BLAS/LAPACK` library. On macOS, install them via `Homebrew`:
    ```bash
    brew install openblas lapack
    ```

3. **Matplotlib**
    Make sure you have `matplotlib` installed. You can install it via pip：
    ```bash
    pip install matplotlib  
    ```

## II、Platforms Build Instructions

The compiler and linker flags (`CFLAGS` and `LDFLAGS`) depend on your system and how OpenBLAS, LAPACK, and MPI are installed. I use a Macbook so the `MAKEFILE` only works for MacOS.

If you want to run the `MAKEFILE`, you should edit the `linker flags` (`CFLAGS` and `LDFLAGS`)  depending on your systems. 
For example, below are the correct configurations for Ubuntu.
```
CFLAGS = -O2 -I/usr/include/
LDFLAGS = -L/usr/lib/ -lopenblas -llapack -lm
```


## III、Building and Running 

1. Run the TSQR program using 4 `MPI` processes:
    ```bash
    make run
    ```
    ![main_result](https://github.com/StarCloudes/case-study/blob/master/assignment01/c-code/main_result.png)
   
   TSQR implementation appears to be working correctly based on the output:
   * The Frobenius norm of the residual is extremely small (~10^(-15)), indicating that QR closely reconstructs A.
   * The Q matrix is orthogonal, as expected. The values in the identity matrix are very close to exact ones, with only minor floating-point precision effects (~10^(-15)).

3. Run the TSQR Performance Test using 4 `MPI` processes:
    ```bash
    make run_perf
    ```
     ![perf_result](https://github.com/StarCloudes/case-study/blob/master/assignment01/c-code/perf_result.png)
   
    After running this code, these results have been saved in `scaling_results.csv`.
   


5. Use the `Python` Script to draw some scaling plots from the `scaling_results.csv` (Note: Make sure you have `python3` installed). 
    ```bash
    python3 plot_scaling.py 
    ```
    This script will produce two plots:
	* `scaling_vs_m.png`: Execution time vs. m for different fixed values of n.
	* `scaling_vs_n.png`: Execution time vs. n for different fixed values of m.


6. To clean all compiled files for the `C/MPI` version, run:
    ```bash
    make clean
    ```

##  IV、Scaling plots

![Scaling vs m](https://github.com/StarCloudes/case-study/blob/master/assignment01/c-code/scaling_vs_m.png)

### 1. Scaling with m: 
The lines for n=8,16,32 eventually show increasing execution time with m. However, there are a few irregular dips or spikes (for example, m=1000 vs. m=2000 in some curves) where the execution time unexpectedly decreases or increases because of overhead for Small Problems.

![Scaling vs m](https://github.com/StarCloudes/case-study/blob/master/assignment01/c-code/scaling_vs_n.png)

### 2. Scaling with n: 
For larger m (like 5000 and 10000), we see that increasing n leads to a clear rise in execution time. For example, at m=10000, time grows from approx 0.00036 s at n=4 to approx 0.01789 s at n=32, which is a substantial jump.
For smaller m (e.g., m=1000 or m=2000), you see some erratic behavior (like a time of 0.0060 s at n=4 but then 0.00044 s at n=8), again likely due to overhead overshadowing the small actual computation.

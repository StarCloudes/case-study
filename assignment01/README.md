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

1. **MPI**  
    Ensure you have an MPI implementation installed (e.g., MPICH, Open MPI). On macOS, you can install Open MPI via Homebrew:
    ```bash
    brew install open-mpi
    ```

2. **OpenBLAS / LAPACK**:
    You need a BLAS/LAPACK library. On macOS, install them via Homebrew:
    ```bash
    brew install openblas lapack
    ```

3. **Matplotlib / numpy**
    Make sure you have matplotlib and numpy installed. You can install it via pip：
    ```bash
    pip3 install matplotlib numpy  
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
    After running this code, these results have been saved in scaling_results.csv CSV file .

3. Use the Python Script to draw some scaling plots from the scaling_results.csv (Note: Here I use python3). 
    ```bash
    python3 plot_scaling.py 
    ```
    This script will produce two plots:
	•	scaling_vs_m.png: Execution time vs. m for different fixed values of n.
	•	scaling_vs_n.png: Execution time vs. n for different fixed values of m.

4. To clean all compiled files for the C/MPI version, run:
    ```bash
    make clean
    ```

##  Test results for different values of m and n

![Scaling vs m](https://github.com/StarCloudes/case-study/blob/master/assignment01/c-code/scaling_vs_m.png)
Scaling with m: The lines for n=8,16,32 eventually show increasing execution time with m. However, there are a few irregular dips or spikes (for example, m=1000 vs. m=2000 in some curves) where the execution time unexpectedly decreases or increases because of overhead for Small Problems.

![Scaling vs m](https://github.com/StarCloudes/case-study/blob/master/assignment01/c-code/scaling_vs_n.png)
Scaling with n: For larger m (like 5000 and 10000), we see that increasing n leads to a clear rise in execution time. For example, at m=10000, time grows from \approx 0.00036 s at n=4 to \approx 0.01789 s at n=32, which is a substantial jump.
For smaller m (e.g., m=1000 or m=2000), you see some erratic behavior (like a time of 0.0060 s at n=4 but then 0.00044 s at n=8), again likely due to overhead overshadowing the small actual computation..

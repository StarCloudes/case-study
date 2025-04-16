# Case Study 3 Conjugate Gradient (CG) algorithm

## Overview
This case study focuses on implementing and analyzing the Conjugate Gradient (CG) algorithm for solving symmetric positive definite linear systems, including both:

-	**Structured matrix (Poisson equation** :
 	Solved using a 5-point finite difference scheme and matrix-free CG method in C.
 	
-	**PDense matrix with known condition number:**: 
 	 Used to analyze convergence behavior and compare against theoretical error bounds.
Both cases involve collecting iteration counts, execution time, and residuals, with optional visualizations in Python.


## File Descriptions 
- **cg_poisson.c**  
  Implements the matrix-free CG solver for 2D Poisson problem using 5-point stencil. Outputs performance results and optional solution profile.

- **cg_convergence_dense.c**  
  Dense CG solver applied to a symmetric Toeplitz matrix with known condition number. Tracks convergence and final residual for large \(N\).

- **residuals_plot.py**  
  Python script to visualize residual norm and theoretical bound over iterations (semilog plot).

- **u_plot.py**  
  Python script to plot the computed solution \(u(x)\) of 3D solution heatmap from exported CSV.

- **CG_Writeup.pdf**  
  PDF writeup for the assignment results and analysis (if generated).
  

## Usage Instructions
1. **Compile the codes**:
 use `make` to compile all the C code files.
2.	**Run Programs**:
 `./cg_poisson`         # for Poisson CG solution
 `./cg_dense`           # for dense CG
3.	**Visualize Results (Python)**
Make sure numpy, matplotlib, and pandas are installed.
`python3 u_plot.py`             # for Poisson CG solution
`python3 residuals_plot.py`     # for dense CG


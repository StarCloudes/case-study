# GMRES Case Study 2

## Overview
This case study aims to implement and verify two versions of the GMRES algorithm:
-	**Serial GMRES** :
  Uses the Arnoldi process to build the Krylov subspace and updates the approximate solution via a least squares solve.
-	**Parallel GMRES**: 
  Parallelizes the inner product computations in the Arnoldi process (using MATLAB’s parfor) to improve efficiency for large-scale problems, while also incorporating an orthonormality check to ensure numerical stability.

## File Descriptions
-	**assignmetn02.mlx**
  - Implements the serial GMRES algorithm using the Arnoldi process and least squares solve, outputting the approximate solution and residual history.
  - Implements the parallel GMRES algorithm using MATLAB’s parfor to accelerate inner product computations, with additional checks for orthonormality and residual accuracy.
-	**plots**
  Plot the relative residual versus iteration number on a semilog graph for GMRES algorithm.
-	**writeup.pdf**
  a summary pdf containing a description of solutions
Plot the relative residual versus iteration number on a semilog graph.
-	**README.md**
  This file, which provides an overall description of the case study, file structure, and usage instructions.

## Usage Instructions
1. **Environment Requirements**:
  MATLAB (preferably R2018b or later) with the Parallel Computing Toolbox (for parfor).
2.	**Running Codes and Tests**:
  Import assignmetn02.mlx to MATLAB and run the whole code to get the results
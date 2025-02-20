/**
 * @file tsqr_refactored.c
 * @brief TSQR decomposition using MPI and LAPACK with detailed step-by-step instructions.
 *
 * This program performs a TSQR (Tall Skinny QR) decomposition on a tall-skinny matrix
 * (e.g., 1000×4) distributed across multiple processes and verifies the result.
 *
 * TSQR Steps:
 *   Step 1: Perform local QR factorization on each block.
 *   Step 2: Merge all R_blocks (stack them vertically).
 *   Step 3: Perform QR factorization on the merged R_combined.
 *   Step 4: Compute the global Q matrix.
 *
 * Note: This example assumes that the number of rows M is divisible by NUM_BLOCKS.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <mpi.h>
 #include <math.h>
 #include <lapacke.h>
 
 #define M 1000        // Number of rows in matrix A
 #define N 4           // Number of columns in matrix A
 #define NUM_BLOCKS 4  // Number of MPI processes
 
 // Generate a random matrix with values in the range [-1, 1]
 void generate_random_matrix(double *A, int m, int n) {
     for (int i = 0; i < m * n; i++) {
         A[i] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
     }
 }
 
 // Print the matrix with a given name
 void print_matrix(double *A, int m, int n, const char *name) {
     printf("%s:\n", name);
     for (int i = 0; i < m; i++) {
         for (int j = 0; j < n; j++) {
             printf("%8.4f ", A[i * n + j]);
         }
         printf("\n");
     }
     printf("\n");
 }
 
 // Perform simple matrix multiplication: C = A (m x k) * B (k x n)
 void matrix_multiply(double *A, double *B, double *C, int m, int k, int n) {
     for (int i = 0; i < m; i++) {
         for (int j = 0; j < n; j++) {
             C[i * n + j] = 0.0;
             for (int l = 0; l < k; l++) {
                 C[i * n + j] += A[i * k + l] * B[l * n + j];
             }
         }
     }
 }
 
 // Compute the Frobenius norm of a matrix
 double frobenius_norm(double *A, int m, int n) {
     double norm = 0.0;
     for (int i = 0; i < m * n; i++) {
         norm += A[i] * A[i];
     }
     return sqrt(norm);
 }
 
 int main(int argc, char **argv) {
     int rank, size;
     MPI_Init(&argc, &argv);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
 
     // Ensure the program is run with the correct number of processes
     if (size != NUM_BLOCKS) {
         if (rank == 0) {
             printf("Error: This program must run with %d processes.\n", NUM_BLOCKS);
         }
         MPI_Finalize();
         return EXIT_FAILURE;
     }
 
     // Check that M is divisible by NUM_BLOCKS
     if (M % NUM_BLOCKS != 0) {
         if (rank == 0) {
             printf("Error: M (%d) must be divisible by NUM_BLOCKS (%d).\n", M, NUM_BLOCKS);
         }
         MPI_Finalize();
         return EXIT_FAILURE;
     }
 
     int block_size = M / NUM_BLOCKS;
     double *A_block = (double *)malloc(block_size * N * sizeof(double));
     
     double *A_full = NULL;
     if (rank == 0) {
         A_full = (double *)malloc(M * N * sizeof(double));
         generate_random_matrix(A_full, M, N);
         // Optionally, print the original matrix (for small matrices)
         // print_matrix(A_full, M, N, "Original Matrix A");
     }
 
     // Scatter the full matrix A into blocks across all processes
     MPI_Scatter(A_full, block_size * N, MPI_DOUBLE,
                 A_block, block_size * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
 
     /*******************************
      * Step 1: Local QR Factorization
      *******************************/
     // Each process computes the QR factorization of its local block:
     // A_block = Q_local * R_local
     double *tau_local = (double *)malloc(N * sizeof(double));
     LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, block_size, N, A_block, N, tau_local);
 
     // Generate the local Q matrix from the QR factorization
     double *Q_local = (double *)malloc(block_size * N * sizeof(double));
     for (int i = 0; i < block_size * N; i++) {
         Q_local[i] = A_block[i];
     }
     LAPACKE_dorgqr(LAPACK_ROW_MAJOR, block_size, N, N, Q_local, N, tau_local);
 
     // Extract the upper-triangular part from A_block as the local R matrix (R_local)
     double *R_local = (double *)malloc(N * N * sizeof(double));
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             R_local[i * N + j] = (i <= j) ? A_block[i * N + j] : 0.0;
         }
     }
 
     /***************************************
      * Step 2: Merge All R_blocks Vertically
      ***************************************/
     // Gather all local R matrices into a combined matrix at the root.
     // R_combined dimensions: (NUM_BLOCKS * N) x N.
     double *R_combined = NULL;
     if (rank == 0) {
         R_combined = (double *)malloc(NUM_BLOCKS * N * N * sizeof(double));
     }
     MPI_Gather(R_local, N * N, MPI_DOUBLE,
                R_combined, N * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
 
     /**************************************************
      * Step 3: Global QR Factorization on Merged R_combined
      **************************************************/
     // The root process performs a QR factorization on R_combined.
     double *R_global = (double *)malloc(N * N * sizeof(double));
     // Q_small has dimensions (NUM_BLOCKS * N) x N and will help adjust Q.
     double *Q_small = (double *)malloc(NUM_BLOCKS * N * N * sizeof(double));
     if (rank == 0) {
         double *tau_global = (double *)malloc(N * sizeof(double));
         LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, NUM_BLOCKS * N, N, R_combined, N, tau_global);
 
         // Extract the upper-triangular matrix as R_global
         for (int i = 0; i < N; i++) {
             for (int j = 0; j < N; j++) {
                 R_global[i * N + j] = (i <= j) ? R_combined[i * N + j] : 0.0;
             }
         }
         // Generate Q_small from the QR factorization of R_combined
         LAPACKE_dorgqr(LAPACK_ROW_MAJOR, NUM_BLOCKS * N, N, N, R_combined, N, tau_global);
         for (int i = 0; i < NUM_BLOCKS * N * N; i++) {
             Q_small[i] = R_combined[i];
         }
         free(tau_global);
     }
     // Broadcast R_global and Q_small to all processes
     MPI_Bcast(R_global, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     MPI_Bcast(Q_small, NUM_BLOCKS * N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
     /********************************************
      * Step 4: Compute the Global Q Matrix
      ********************************************/
     // Gather local Q matrices into Q_global at the root.
     double *Q_global = NULL;
     if (rank == 0) {
         Q_global = (double *)malloc(M * N * sizeof(double));
     }
     MPI_Gather(Q_local, block_size * N, MPI_DOUBLE,
                Q_global, block_size * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
 
     if (rank == 0) {
         // Multiply each local Q block with the corresponding Q_small block to form the final Q matrix.
         double *Q_final = (double *)malloc(M * N * sizeof(double));
         for (int i = 0; i < NUM_BLOCKS; i++) {
             matrix_multiply(Q_global + i * block_size * N,
                             Q_small + i * N * N,
                             Q_final + i * block_size * N,
                             block_size, N, N);
         }
         
         // --------------------------
         // Verification of the Result
         // --------------------------
         // Compute QR = Q_final * R_global and compare with the original A.
         double *QR = (double *)malloc(M * N * sizeof(double));
         matrix_multiply(Q_final, R_global, QR, M, N, N);
 
         // Calculate the residual ||QR - A||_F
         double *A_minus_QR = (double *)malloc(M * N * sizeof(double));
         for (int i = 0; i < M * N; i++) {
             A_minus_QR[i] = A_full[i] - QR[i];
         }
         double error = frobenius_norm(A_minus_QR, M, N);
         printf("||QR - A||_F = %.5e\n", error);
 
         // Verify the orthogonality of Q_final by computing Q_final^T * Q_final
         double *Q_transpose = (double *)malloc(N * M * sizeof(double));
         for (int i = 0; i < M; i++) {
             for (int j = 0; j < N; j++) {
                 Q_transpose[j * M + i] = Q_final[i * N + j];
             }
         }
         double *QtQ = (double *)malloc(N * N * sizeof(double));
         matrix_multiply(Q_transpose, Q_final, QtQ, N, M, N);
         print_matrix(QtQ, N, N, "Q^T * Q (should be identity)");
 
         // Free memory on the root process
         free(Q_final);
         free(QR);
         free(A_minus_QR);
         free(Q_transpose);
         free(QtQ);
         free(Q_global);
     }
 
     // Free memory allocated on all processes
     free(A_block);
     free(R_local);
     free(tau_local);
     free(Q_local);
     free(R_combined);
     free(R_global);
     free(Q_small);
     if (rank == 0) {
         free(A_full);
     }
 
     MPI_Finalize();
     return EXIT_SUCCESS;
 }
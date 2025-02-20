#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include "matrix_utils.h"
#include "tsqr_module.h"

#define M 1000        // Total number of rows in matrix A
#define N 4           // Number of columns in matrix A
#define NUM_BLOCKS 4  // Number of MPI processes (and blocks)

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if we're running with the required number of processes
    if (size != NUM_BLOCKS) {
        if (rank == 0) {
            printf("Error: This program must run with %d processes.\n", NUM_BLOCKS);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Ensure that M can be evenly divided by NUM_BLOCKS
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
        // Optionally, print the original matrix (for small sizes)
        // print_matrix(A_full, M, N, "Original Matrix A");
    }

    // Distribute the matrix A among the processes
    MPI_Scatter(A_full, block_size * N, MPI_DOUBLE,
                A_block, block_size * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /***********************
     * Step 1: Local QR
     ***********************/
    double *tau_local = (double *)malloc(N * sizeof(double));
    local_qr_decomposition(A_block, block_size, N, tau_local);

    double *Q_local = (double *)malloc(block_size * N * sizeof(double));
    generate_q_local(A_block, block_size, N, tau_local, Q_local);

    double *R_local = (double *)malloc(N * N * sizeof(double));
    extract_r_local(A_block, block_size, N, R_local);

    /*******************************
     * Step 2: Merge R Blocks
     *******************************/
    double *R_combined = NULL;
    if (rank == 0) {
        R_combined = (double *)malloc(NUM_BLOCKS * N * N * sizeof(double));
    }
    MPI_Gather(R_local, N * N, MPI_DOUBLE,
               R_combined, N * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    /******************************************
     * Step 3: Global QR on Combined R Blocks
     ******************************************/
    double *R_global = (double *)malloc(N * N * sizeof(double));
    double *Q_small = (double *)malloc(NUM_BLOCKS * N * N * sizeof(double));
    if (rank == 0) {
        global_qr_decomposition(R_combined, NUM_BLOCKS, N, R_global, Q_small);
    }
    MPI_Bcast(R_global, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Q_small, NUM_BLOCKS * N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /************************************
     * Step 4: Compute the Global Q
     ************************************/
    double *Q_global = NULL;
    if (rank == 0) {
        Q_global = (double *)malloc(M * N * sizeof(double));
    }
    MPI_Gather(Q_local, block_size * N, MPI_DOUBLE,
               Q_global, block_size * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        double *Q_final = (double *)malloc(M * N * sizeof(double));
        compute_final_q(Q_global, Q_small, Q_final, block_size, NUM_BLOCKS, N);

        // Verify the TSQR result: compute Q_final * R_global and compare with the original A
        double *QR = (double *)malloc(M * N * sizeof(double));
        matrix_multiply(Q_final, R_global, QR, M, N, N);

        double *A_minus_QR = (double *)malloc(M * N * sizeof(double));
        for (int i = 0; i < M * N; i++) {
            A_minus_QR[i] = A_full[i] - QR[i];
        }
        double error = frobenius_norm(A_minus_QR, M, N);
        printf("||QR - A||_F = %.5e\n", error);

        // Check the orthogonality of Q_final: compute Q_final^T * Q_final (should be close to identity)
        double *Q_transpose = (double *)malloc(N * M * sizeof(double));
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                Q_transpose[j * M + i] = Q_final[i * N + j];
            }
        }
        double *QtQ = (double *)malloc(N * N * sizeof(double));
        matrix_multiply(Q_transpose, Q_final, QtQ, N, M, N);
        print_matrix(QtQ, N, N, "Q^T * Q (should be identity)");

        free(Q_final);
        free(QR);
        free(A_minus_QR);
        free(Q_transpose);
        free(QtQ);
        free(Q_global);
    }

    // Free all allocated memory
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
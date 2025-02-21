#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include "matrix_utils.h"
#include "tsqr_module.h"

/**
 * @brief Main function to perform performance scalability tests.
 *
 * This program loops over a series of test matrix sizes (m x n), runs the TSQR
 * decomposition on each, and measures the execution time. The timing results for each
 * test case are written as CSV lines to "scaling_results.csv" (on the root process).
 *
 * Note: m must be divisible by the number of MPI processes.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return EXIT_SUCCESS if all tests complete successfully.
 */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int num_blocks = size;  // we assume one block per process

    // Define arrays for test matrix sizes.
    // Make sure m values are multiples of num_blocks.
    int m_tests[] = {1000, 2000, 5000, 10000};
    int n_tests[] = {4, 8, 16, 32};
    int num_m_tests = sizeof(m_tests) / sizeof(m_tests[0]);
    int num_n_tests = sizeof(n_tests) / sizeof(n_tests[0]);

    // Only rank 0 will create and write to the CSV file.
    FILE *csv_file = NULL;
    if (rank == 0) {
        csv_file = fopen("scaling_results.csv", "w");
        if (!csv_file) {
            fprintf(stderr, "Failed to open scaling_results.csv for writing.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        // Write CSV header.
        fprintf(csv_file, "m,n,execution_time_seconds\n");
    }

    // Loop over each combination of m and n.
    for (int i = 0; i < num_m_tests; i++) {
        for (int j = 0; j < num_n_tests; j++) {
            int m = m_tests[i];
            int n = n_tests[j];

            // Skip if m is not divisible by the number of processes.
            if (m % num_blocks != 0) {
                if (rank == 0) {
                    printf("Skipping test (m=%d, n=%d) because m is not divisible by %d\n", m, n, num_blocks);
                }
                continue;
            }
            int block_size = m / num_blocks;

            // Root generates the full m x n matrix.
            double *A_full = NULL;
            if (rank == 0) {
                A_full = (double *)malloc(m * n * sizeof(double));
                generate_random_matrix(A_full, m, n);
            }

            // Each process allocates space for its block.
            double *A_block = (double *)malloc(block_size * n * sizeof(double));

            // Scatter the full matrix among processes.
            MPI_Scatter(A_full, block_size * n, MPI_DOUBLE,
                        A_block, block_size * n, MPI_DOUBLE,
                        0, MPI_COMM_WORLD);

            // Synchronize all processes and start timing.
            MPI_Barrier(MPI_COMM_WORLD);
            double start_time = MPI_Wtime();

            /* TSQR Steps */
            // Step 1: Local QR factorization
            double *tau_local = (double *)malloc(n * sizeof(double));
            local_qr_decomposition(A_block, block_size, n, tau_local);

            double *Q_local = (double *)malloc(block_size * n * sizeof(double));
            generate_q_local(A_block, block_size, n, tau_local, Q_local);

            double *R_local = (double *)malloc(n * n * sizeof(double));
            extract_r_local(A_block, block_size, n, R_local);

            // Step 2: Merge local R matrices
            double *R_combined = NULL;
            if (rank == 0) {
                R_combined = (double *)malloc(num_blocks * n * n * sizeof(double));
            }
            MPI_Gather(R_local, n * n, MPI_DOUBLE,
                       R_combined, n * n, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);

            // Step 3: Global QR on combined R blocks
            double *R_global = (double *)malloc(n * n * sizeof(double));
            double *Q_small = (double *)malloc(num_blocks * n * n * sizeof(double));
            if (rank == 0) {
                global_qr_decomposition(R_combined, num_blocks, n, R_global, Q_small);
            }
            MPI_Bcast(R_global, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(Q_small, num_blocks * n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Step 4: Compute the global Q matrix.
            double *Q_global = NULL;
            if (rank == 0) {
                Q_global = (double *)malloc(m * n * sizeof(double));
            }
            MPI_Gather(Q_local, block_size * n, MPI_DOUBLE,
                       Q_global, block_size * n, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);

            double *Q_final = NULL;
            if (rank == 0) {
                Q_final = (double *)malloc(m * n * sizeof(double));
                compute_final_q(Q_global, Q_small, Q_final, block_size, num_blocks, n);
            }

            // Synchronize all processes and stop timing.
            MPI_Barrier(MPI_COMM_WORLD);
            double end_time = MPI_Wtime();
            double exec_time = end_time - start_time;

            // On rank 0, output the result for this test to CSV.
            if (rank == 0) {
                fprintf(csv_file, "%d,%d,%.8f\n", m, n, exec_time);
                printf("Test (m=%d, n=%d): execution time = %.8f seconds\n", m, n, exec_time);
            }

            // Free allocated memory for this test.
            free(A_block);
            free(tau_local);
            free(Q_local);
            free(R_local);
            free(R_global);
            free(Q_small);
            if (rank == 0) {
                free(A_full);
                free(Q_global);
                free(Q_final);
                free(R_combined);
            }
        }
    }

    // Close the CSV file on rank 0.
    if (rank == 0 && csv_file) {
        fclose(csv_file);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
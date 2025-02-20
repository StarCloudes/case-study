#include "tsqr_module.h"
#include <lapacke.h>
#include <string.h>
#include "matrix_utils.h"  // For matrix_multiply

/**
 * @brief Performs a local QR factorization on a matrix block.
 *
 * Uses LAPACKE_dgeqrf to compute the QR factorization of A_block.
 *
 * @param A_block Pointer to the matrix block.
 * @param block_size Number of rows in the matrix block.
 * @param N Number of columns in the matrix block.
 * @param tau_local Output array for tau values.
 */
void local_qr_decomposition(double *A_block, int block_size, int N, double *tau_local) {
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, block_size, N, A_block, N, tau_local);
}

/**
 * @brief Extracts the upper triangular matrix R from the QR factorization.
 *
 * This function copies the upper triangular portion of A_block into R_local.
 *
 * @param A_block Pointer to the matrix block after QR factorization.
 * @param block_size Number of rows in the matrix block.
 * @param N Number of columns in the matrix block.
 * @param R_local Output matrix for the upper triangular R.
 */
void extract_r_local(double *A_block, int block_size, int N, double *R_local) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            R_local[i * N + j] = (i <= j) ? A_block[i * N + j] : 0.0;
        }
    }
}

/**
 * @brief Generates the local Q matrix from the QR factorization.
 *
 * Copies A_block into Q_local and calls LAPACKE_dorgqr to generate the orthogonal matrix Q.
 *
 * @param A_block Pointer to the matrix block after QR factorization.
 * @param block_size Number of rows in the matrix block.
 * @param N Number of columns in the matrix block.
 * @param tau_local Array of tau values from the QR factorization.
 * @param Q_local Output matrix for the local Q.
 */
void generate_q_local(double *A_block, int block_size, int N, double *tau_local, double *Q_local) {
    memcpy(Q_local, A_block, block_size * N * sizeof(double));
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, block_size, N, N, Q_local, N, tau_local);
}

/**
 * @brief Performs a global QR factorization on the combined R matrices.
 *
 * Computes a QR factorization on the vertically stacked local R matrices and extracts the global R (upper triangular)
 * as well as the helper matrix Q_small.
 *
 * @param R_combined Pointer to the combined R matrices.
 * @param num_blocks Number of local R blocks.
 * @param N Number of columns in the matrices.
 * @param R_global Output global R matrix.
 * @param Q_small Output helper matrix Q_small.
 */
void global_qr_decomposition(double *R_combined, int num_blocks, int N, double *R_global, double *Q_small) {
    int rows = num_blocks * N;
    double *tau_global = (double *)malloc(N * sizeof(double));
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, rows, N, R_combined, N, tau_global);

    // Extract the upper triangular matrix as R_global
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            R_global[i * N + j] = (i <= j) ? R_combined[i * N + j] : 0.0;
        }
    }

    // Generate the helper matrix Q_small
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rows, N, N, R_combined, N, tau_global);
    memcpy(Q_small, R_combined, rows * N * sizeof(double));
    free(tau_global);
}

/**
 * @brief Computes the final global Q matrix.
 *
 * Multiplies each local Q block by the corresponding block in Q_small to form the final global Q.
 *
 * @param Q_global Pointer to the gathered local Q matrices.
 * @param Q_small Pointer to the helper matrix Q_small.
 * @param Q_final Output final global Q matrix.
 * @param block_size Number of rows per local Q matrix.
 * @param num_blocks Number of local Q blocks.
 * @param N Number of columns in the Q matrices.
 */
void compute_final_q(double *Q_global, double *Q_small, double *Q_final, int block_size, int num_blocks, int N) {
    for (int i = 0; i < num_blocks; i++) {
        matrix_multiply(Q_global + i * block_size * N,
                        Q_small + i * N * N,
                        Q_final + i * block_size * N,
                        block_size, N, N);
    }
}
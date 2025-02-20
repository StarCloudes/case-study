#include "matrix_utils.h"

/**
 * @brief Generates a random matrix with values between -1 and 1.
 *
 * This function fills an m x n matrix with random numbers in the range [-1, 1].
 *
 * @param A Pointer to the matrix (stored as a 1D array in row-major order) to be filled.
 * @param m The number of rows in the matrix.
 * @param n The number of columns in the matrix.
 */
void generate_random_matrix(double *A, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        A[i] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

/**
 * @brief Prints the contents of a matrix.
 *
 * This function prints an m x n matrix to the standard output.
 *
 * @param A Pointer to the matrix (stored as a 1D array in row-major order).
 * @param m The number of rows in the matrix.
 * @param n The number of columns in the matrix.
 * @param name A string representing the name of the matrix.
 */
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

/**
 * @brief Multiplies two matrices.
 *
 * This function computes the product C = A * B, where A is an m x k matrix and B is a k x n matrix.
 *
 * @param A Pointer to the first matrix (of size m x k).
 * @param B Pointer to the second matrix (of size k x n).
 * @param C Pointer to the result matrix (of size m x n). The memory for C must be allocated by the caller.
 * @param m The number of rows in matrix A.
 * @param k The number of columns in matrix A and the number of rows in matrix B.
 * @param n The number of columns in matrix B.
 */
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

/**
 * @brief Computes the Frobenius norm of a matrix.
 *
 * The Frobenius norm is the square root of the sum of the squares of all elements in the matrix.
 *
 * @param A Pointer to the matrix (stored as a 1D array in row-major order).
 * @param m The number of rows in the matrix.
 * @param n The number of columns in the matrix.
 * @return The Frobenius norm of the matrix.
 */
double frobenius_norm(double *A, int m, int n) {
    double norm = 0.0;
    for (int i = 0; i < m * n; i++) {
        norm += A[i] * A[i];
    }
    return sqrt(norm);
}
#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* Create an m x n matrix filled with random numbers between -1 and 1 */
void generate_random_matrix(double *A, int m, int n);

/* Print out a matrix */
void print_matrix(double *A, int m, int n, const char *name);

/* Multiply two matrices: C = A (m x k) * B (k x n) */
void matrix_multiply(double *A, double *B, double *C, int m, int k, int n);

/* Compute the Frobenius norm of a matrix */
double frobenius_norm(double *A, int m, int n);

#endif // MATRIX_UTILS_H
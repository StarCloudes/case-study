#ifndef TSQR_MODULE_H
#define TSQR_MODULE_H

/* Performs a local QR factorization on a matrix block. */
void local_qr_decomposition(double *A_block, int block_size, int N, double *tau_local);

/* Extracts the upper triangular matrix (R) from the QR factorization */
void extract_r_local(double *A_block, int block_size, int N, double *R_local);

/* Generates the local Q matrix from a QR factorization. */
void generate_q_local(double *A_block, int block_size, int N, double *tau_local, double *Q_local);

/* Performs a global QR factorization on combined R matrices. */
void global_qr_decomposition(double *R_combined, int num_blocks, int N, double *R_global, double *Q_small);

/* Computes the final global Q matrix */
void compute_final_q(double *Q_global, double *Q_small, double *Q_final, int block_size, int num_blocks, int N);

#endif // TSQR_H
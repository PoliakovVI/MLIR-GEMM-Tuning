/* C source code is found in dgemm_example.c */

#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "mkl.h"

using std::min;

extern "C" void printFlops(double t);
extern "C" double rtclock();

int main()
{
    float *A, *B, *C;
    int m, n, k, i, j, num_reps;
    float alpha, beta;

    // printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
    //         " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
    //         " alpha and beta are float precision scalars\n\n");

    m = 1024, k = 1024, n = 1024, num_reps = 10;
    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0; beta = 1.0;

    // printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
    //         " performance \n\n");
    A = (float *)mkl_malloc( m*k*sizeof( float ), 64 );
    B = (float *)mkl_malloc( k*n*sizeof( float ), 64 );
    C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }

    // printf (" Intializing matrix data \n\n");
    for (i = 0; i < (m*k); i++) {
        A[i] = (float)(1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (float)(1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 1.0;
    }

    // printf (" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    double t_start = rtclock();
    for (i = 0; i < num_reps; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
    }
    double t_end = rtclock();
    // printf ("\n Computations completed.\n\n");

    // printf (" Top left corner of matrix A: \n");
    // for (i=0; i< min(m,6); i++) {
    //   for (j=0; j< min(k,6); j++) {
    //     printf ("%12.0f", A[j+i*k]);
    //   }
    //   printf ("\n");
    // }

    // printf ("\n Top left corner of matrix B: \n");
    // for (i=0; i< min(k,6); i++) {
    //   for (j=0; j< min(n,6); j++) {
    //     printf ("%12.0f", B[j+i*n]);
    //   }
    //   printf ("\n");
    // }
    
    // printf ("\n Top left corner of matrix C: \n");
    // for (i=0; i< min(m,6); i++) {
    //   for (j=0; j< min(n,6); j++) {
    //     printf ("%12.5G", C[j+i*n]);
    //   }
    //   printf ("\n");
    // }

    printf ("Element: %1.5G\n\n", C[0]);

    // printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    // num_flops_total = 2 * M * N * K * num_reps
    double flops_per_s = ((int64_t)2 * m * n * k * num_reps) / (t_end - t_start);
    
    printFlops(flops_per_s);

    // printf (" Example completed. \n\n");
    return 0;
}
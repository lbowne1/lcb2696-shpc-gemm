#include "assignment3.h"
#include <immintrin.h>
#include <assert.h>

int MR = 4;
int NR = 4;

int KC = 64;
int MC = 16;
int NC = 8;

void shpc_dgemm(int m, int n, int k,
                double *A, int rsA, int csA,
                double *B, int rsB, int csB,
                double *C, int rsC, int csC)
{

    double *Ac = (double *)malloc(MC * KC * sizeof(double));
    double *Bc = (double *)malloc(NC * KC * sizeof(double));

    // First loop partitions C and B into column panels.
    for (int jc = 0; jc < n; jc += NC)
    {

        // Second loop partitions A and B over KC
        for (int pc = 0; pc < k; pc += KC)
        {
            PackPanelB_KCxNC(KC, NC, B + jc * csB + pc * rsB, csB, rsB, Bc);
            /* //
            for (int np = 0; np < NC; np += NR)
                for (int pp = 0; pp < KC; pp++) {
                    for (int jp = 0; jp < NR; jp++) {
                        Bc[(np + jp + pp * KC)] = B[(pp) * rsB + (pc + jc + j) * csB];
                    }
                } */
            // Pack B
            /* for (int n = 0; n < NC; n += NR)
            {
                for (int p = 0; p < KC; p++)
                {
                    for (int j = 0; j < NR; j++)
                    {
                        //Bc[n + p * NR + j] = B[(n + j) * csB + (p + pc) * rsB];
                        //Bc[(n + j) * NC + p * 1] = B[(jc * csB) + (pc * rsB) + (n * KC) * NR + p * NR + j];
                        Bc[p * NR + j + n] = B[(pc + p) * rsB + (jc + n + j) * csB];
                    }
                }
            } */

            // Third loop partitions C and A over MC
            for (int ic = 0; ic < m; ic += MC)
            {
                PackPanelA_MCxKC(MC, KC, A + ic * rsA + pc * csA, rsA, csA, Ac);

                /*
                                for (int p = 0; p < KC; p++)
                                {
                                    for (int i = 0; i < MC; i++)
                                    {
                                        Ac[p * MC + i] = A[(pc + p) * rsA + (ic + i) * csA];
                                    }
                                } */

                // Pack A
                /* for (int m = 0; m < MC; m += MR)
                {
                    for (int p = 0; p < KC; p++)
                    {
                        for (int i = 0; i < MR; i++)
                        {
                            //Ac[(m + p) * MR + i] = A[(m + i) * rsA + p * csA];
                            //Ac[(m + i) * 1 + p * MC] = A[(pc * csA) + (ic + rsA) + (m * KC) * MR + p * MR + i];
                            Ac[p * MR + i + m] = A[(pc + p) * rsA + (ic + m + i) * csA];

                        }
                    }
                } */

                // Fourth loop partitions Bp into column “slivers” of width nr.
                for (int jr = 0; jr < NC; jr += NR)
                {

                    // Fifth loop partitions Ae into row slivers of height mr.
                    for (int ir = 0; ir < MC; ir += MR)
                    {

                        // packed rsx and csx
                        /*
                        microkernel(Ac + ir * 1, 1, MR,
                        Bc + (jc + jr) * 1, NR, 1,
                        C + (ic + ir) * rsC + (jc + jr) * csC, rsC, csC,
                        KC);
                        */

                        microkernel(Ac + (ir * KC), 1, MR,
                                    Bc + (jr * KC), NR, 1,
                                    C + ((ic + ir) * rsC + (jc + jr) * csC), rsC, csC,
                                    KC);

                        /* microkernel(A + (pc * csA) + (ic + ir) * rsA, rsA, csA,
                                    B + (jc + jr) * csB + (pc * rsB), rsB, csB,
                                    C + (ic + ir) * rsC + (jc + jr) * csC, rsC, csC,
                                    KC); */
                    }
                }
            }
        }
        // free(Bc);
        // free(Ac);
    }
}

/*
BLIS micro-kernel
then multiplies the current sliver of Ae by the current sliver
of Be to update the corresponding mr × nr block of C. This
micro-kernel performs a sequence of rank-1 updates (outer
products) with columns from the sliver of Ae and rows from
the sliver of B */
void microkernel(double *A, int rsA, int csA,
                 double *B, int rsB, int csB,
                 double *C, int rsC, int csC,
                 int k)
{
    {
        __m256d gamma_0123_0 = _mm256_loadu_pd(&C[0 * rsC + 0 * csC]);
        __m256d gamma_0123_1 = _mm256_loadu_pd(&C[0 * rsC + 1 * csC]);
        __m256d gamma_0123_2 = _mm256_loadu_pd(&C[0 * rsC + 2 * csC]);
        __m256d gamma_0123_3 = _mm256_loadu_pd(&C[0 * rsC + 3 * csC]);

        /* __m256d gamma_4567_4 = _mm256_loadu_pd(&C[4 * rsC + 0 * csC]);
        __m256d gamma_4567_5 = _mm256_loadu_pd(&C[4 * rsC + 1 * csC]);
        __m256d gamma_4567_6 = _mm256_loadu_pd(&C[4 * rsC + 2 * csC]);
        __m256d gamma_4567_7 = _mm256_loadu_pd(&C[4 * rsC + 3 * csC]); */

        for (int p = 0; p < k; p++)
        {
            // for ( int j=0; j<NR; j++)
            //{
            /* Declare vector registers to hold element j of C and hold it */
            // gamma_0123_j = _mm256_loadu_pd( &C[0*rsC + j*csC] );

            /* Declare vector register for load/broadcasting beta( p,j ) */
            __m256d beta_p_0;
            __m256d beta_p_1;
            __m256d beta_p_2;
            __m256d beta_p_3;
            /* __m256d beta_p_4;
            __m256d beta_p_5;
            __m256d beta_p_6;
            __m256d beta_p_7; */

            /* Declare a vector register to hold the current column of A and load
            it with the four elements of that column. */
            __m256d alpha_0123_p = _mm256_loadu_pd(&A[0 * rsA + p * csA]);
            // __m256d alpha_4567_p = _mm256_loadu_pd(&A[4 * rsA + p * csA]);

            /* Load/broadcast beta( p,0 ). */
            beta_p_0 = _mm256_broadcast_sd(&B[p * rsB + 0 * csB]); // rsB = KC
            beta_p_1 = _mm256_broadcast_sd(&B[p * rsB + 1 * csB]);
            beta_p_2 = _mm256_broadcast_sd(&B[p * rsB + 2 * csB]);
            beta_p_3 = _mm256_broadcast_sd(&B[p * rsB + 3 * csB]);

            /* beta_p_4 = _mm256_broadcast_sd(&B[p * rsB + 4 * csB]);
            beta_p_5 = _mm256_broadcast_sd(&B[p * rsB + 5 * csB]);
            beta_p_6 = _mm256_broadcast_sd(&B[p * rsB + 6 * csB]);
            beta_p_7 = _mm256_broadcast_sd(&B[p * rsB + 7 * csB]); */

            /* update the first column of C with the current column of A times
            beta ( p,0 ) */
            gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_p_0, gamma_0123_0);
            gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_p_1, gamma_0123_1);
            gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_p_2, gamma_0123_2);
            gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_p_3, gamma_0123_3);

            /* gamma_4567_4 = _mm256_fmadd_pd(alpha_4567_p, beta_p_4, gamma_4567_4);
            gamma_4567_5 = _mm256_fmadd_pd(alpha_4567_p, beta_p_5, gamma_4567_5);
            gamma_4567_6 = _mm256_fmadd_pd(alpha_4567_p, beta_p_6, gamma_4567_6);
            gamma_4567_7 = _mm256_fmadd_pd(alpha_4567_p, beta_p_7, gamma_4567_7); */

            /* Store the updated results */
            //_mm256_storeu_pd( &C[0*rsC + j*csC], gamma_0123_j );

            //}
        }
        _mm256_storeu_pd(&C[0 * rsC + 0 * csC], gamma_0123_0);
        _mm256_storeu_pd(&C[0 * rsC + 1 * csC], gamma_0123_1);
        _mm256_storeu_pd(&C[0 * rsC + 2 * csC], gamma_0123_2);
        _mm256_storeu_pd(&C[0 * rsC + 3 * csC], gamma_0123_3);

        /* _mm256_storeu_pd(&C[0 * rsC + 0 * csC], gamma_4567_4);
        _mm256_storeu_pd(&C[0 * rsC + 1 * csC], gamma_4567_5);
        _mm256_storeu_pd(&C[0 * rsC + 2 * csC], gamma_4567_6);
        _mm256_storeu_pd(&C[0 * rsC + 3 * csC], gamma_4567_7); */
    }
}

void PackPanelB_KCxNC(int k, int n, double *B, int csB, int rsB, double *Bc)
/* Pack a k x n panel of B in to a KC x NC buffer.
The block is copied into Btilde a micro-panel at a time. */
{
    for (int jp = 0; jp < n; jp += NR)
    {
        // int jb = bli_min(NR, n - j);

        for (int p = 0; p < k; p++)
        {
            for (int j = 0; j < NR; j++)
            {
                *(Bc++) = B[(jp + j) * csB + p * rsB];
            }
        }
        // PackMicroPanelB_KCxNR(k, NR, &B[j * csB], csB, Btilde);
        // Btilde += k * NR;
    }
}

void PackPanelA_MCxKC(int m, int k, double *A, int rsA, int csA, double *Ac)
/* Pack a m x k panel of A into a MC x KC buffer.
   - The block is copied into Atilde a micro-panel at a time.
   - A is stored in row-major order.
   - The packed format is MC x KC, where MR-sized micro-panels are processed.
*/
{
    for (int ip = 0; ip < m; ip += MR)
    { // Loop over rows in MR-sized chunks
        // int ib = bli_min(MR, m - i); // Handle remainder cases when MC isn't a multiple of MR

        for (int p = 0; p < k; p++)
        {
            for (int i = 0; i < MR; i++)
            {
                *(Ac++) = A[(ip + i) * rsA + p * csA]; // Copy A in row-major format into packed buffer
            }
        }

        // PackMicroPanelA_MRxKC( ib, k, &A[i], ldA, Atilde );
        // Atilde += ib * k; // Move to next micro-panel position in Atilde
    }
}

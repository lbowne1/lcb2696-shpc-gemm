#include "assignment3.h"
#include <immintrin.h>
#include <assert.h>
#include <omp.h>


int MR = 8;
int NR = 6;

int KC = 256;
int MC = 968;
int NC = 11262;

void shpc_dgemm(int m, int n, int k,
                double *A, int rsA, int csA,
                double *B, int rsB, int csB,
                double *C, int rsC, int csC)
{

    double *Ac = (double *)_mm_malloc(MC * KC * sizeof(double), 64);
    double *Bc = (double *)_mm_malloc(NC * KC * sizeof(double), 64);

    // First loop partitions C and B into column panels.
    
        for (int jc = 0; jc < n; jc += NC)
        {
            int real_NC = (NC > (n - jc)) ? (n - jc) : NC;

            // Second loop partitions A and B over KC
            for (int pc = 0; pc < k; pc += KC)
            {
                int real_KC = (KC > (k - pc)) ? (k - pc) : KC;
                pack_panel_b(real_KC, real_NC, B + jc * csB + pc * rsB, csB, rsB, Bc);
                

                // Third loop partitions C and A over MC
                for (int ic = 0; ic < m; ic += MC)
                {
                    int real_MC = (MC < m - ic) ? MC : m - ic;

                    //int max_threads = omp_get_max_threads();
                    //int NC_per_thread = ( ( real_NC / max_threads ) / NR ) * NR;

                    pack_panel_a(real_MC, real_KC, A + ic * rsA + pc * csA, rsA, csA, Ac);

                    // Fourth loop partitions Bp into column “slivers” of width nr.
                    for (int jr = 0; jr < real_NC; jr += NR)
                    {
                        
                        // Fifth loop partitions Ae into row slivers of height mr.
                        for (int ir = 0; ir < real_MC; ir += MR)
                        {

                            microkernel(Ac + (ir * real_KC), 1, MR,
                                        Bc + (jr * real_KC), NR, 1,
                                        C + ((ic + ir) * rsC + (jc + jr) * csC), rsC, csC,
                                        real_KC);
                        }
                    }
                }
            }
        }
    _mm_free(Bc);
    _mm_free(Ac);
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
        __m256d gamma_4567_0 = _mm256_loadu_pd(&C[4 * rsC + 0 * csC]);

        __m256d gamma_0123_1 = _mm256_loadu_pd(&C[0 * rsC + 1 * csC]);
        __m256d gamma_4567_1 = _mm256_loadu_pd(&C[4 * rsC + 1 * csC]);

        __m256d gamma_0123_2 = _mm256_loadu_pd(&C[0 * rsC + 2 * csC]);
        __m256d gamma_4567_2 = _mm256_loadu_pd(&C[4 * rsC + 2 * csC]);

        __m256d gamma_0123_3 = _mm256_loadu_pd(&C[0 * rsC + 3 * csC]);
        __m256d gamma_4567_3 = _mm256_loadu_pd(&C[4 * rsC + 3 * csC]);

        __m256d gamma_0123_4 = _mm256_loadu_pd(&C[0 * rsC + 4 * csC]);
        __m256d gamma_4567_4 = _mm256_loadu_pd(&C[4 * rsC + 4 * csC]);

        __m256d gamma_0123_5 = _mm256_loadu_pd(&C[0 * rsC + 5 * csC]);
        __m256d gamma_4567_5 = _mm256_loadu_pd(&C[4 * rsC + 5 * csC]);


        __m256d beta_p;
        __m256d alpha_0123_p;
        __m256d alpha_4567_p;

        for (int p = 0; p < k; p+=2)
        {
            alpha_0123_p = _mm256_loadu_pd(&A[0 * rsA + p * csA]);
            alpha_4567_p = _mm256_loadu_pd(&A[4 * rsA + p * csA]);

            beta_p = _mm256_broadcast_sd(&B[p * rsB + 0 * csB]); // rsB = KC
            gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_0);
            gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_0);

            beta_p = _mm256_broadcast_sd(&B[p * rsB + 1 * csB]);
            gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_1);
            gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_1);

            beta_p = _mm256_broadcast_sd(&B[p * rsB + 2 * csB]);
            gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_2);
            gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_2);

            beta_p = _mm256_broadcast_sd(&B[p * rsB + 3 * csB]);
            gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_3);
            gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_3);

            beta_p = _mm256_broadcast_sd(&B[p * rsB + 4 * csB]);
            gamma_0123_4 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_4);
            gamma_4567_4 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_4);

            beta_p = _mm256_broadcast_sd(&B[p * rsB + 5 * csB]);
            gamma_0123_5 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_5);
            gamma_4567_5 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_5);


            // Unroll (p + 1) 
            alpha_0123_p = _mm256_loadu_pd(&A[0 * rsA + (p + 1) * csA]);
            alpha_4567_p = _mm256_loadu_pd(&A[4 * rsA + (p + 1) * csA]);

            beta_p = _mm256_broadcast_sd(&B[(p +1) * rsB + 0 * csB]); // rsB = KC
            gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_0);
            gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_0);

            beta_p = _mm256_broadcast_sd(&B[(p +1) * rsB + 1 * csB]);
            gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_1);
            gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_1);

            beta_p = _mm256_broadcast_sd(&B[(p +1) * rsB + 2 * csB]);
            gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_2);
            gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_2);

            beta_p = _mm256_broadcast_sd(&B[(p +1) * rsB + 3 * csB]);
            gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_3);
            gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_3);

            beta_p = _mm256_broadcast_sd(&B[(p +1) * rsB + 4 * csB]);
            gamma_0123_4 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_4);
            gamma_4567_4 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_4);

            beta_p = _mm256_broadcast_sd(&B[(p +1) * rsB + 5 * csB]);
            gamma_0123_5 = _mm256_fmadd_pd(alpha_0123_p, beta_p, gamma_0123_5);
            gamma_4567_5 = _mm256_fmadd_pd(alpha_4567_p, beta_p, gamma_4567_5);
        }

        _mm256_storeu_pd(&C[0 * rsC + 0 * csC], gamma_0123_0);
        _mm256_storeu_pd(&C[4 * rsC + 0 * csC], gamma_4567_0);

        _mm256_storeu_pd(&C[0 * rsC + 1 * csC], gamma_0123_1);
        _mm256_storeu_pd(&C[4 * rsC + 1 * csC], gamma_4567_1);

        _mm256_storeu_pd(&C[0 * rsC + 2 * csC], gamma_0123_2);
        _mm256_storeu_pd(&C[4 * rsC + 2 * csC], gamma_4567_2);

        _mm256_storeu_pd(&C[0 * rsC + 3 * csC], gamma_0123_3);
        _mm256_storeu_pd(&C[4 * rsC + 3 * csC], gamma_4567_3);

        _mm256_storeu_pd(&C[0 * rsC + 4 * csC], gamma_0123_4);
        _mm256_storeu_pd(&C[4 * rsC + 4 * csC], gamma_4567_4);

        _mm256_storeu_pd(&C[0 * rsC + 5 * csC], gamma_0123_5);
        _mm256_storeu_pd(&C[4 * rsC + 5 * csC], gamma_4567_5);
    }
}

/* Pack a k x n panel of B in to a KC x NC buffer.
The block is copied into Btilde a micro-panel at a time. */
void pack_panel_b(int k, int n, double *B, int csB, int rsB, double *Bc)
{
    for (int jp = 0; jp < n; jp += NR)
    {
        int real_NR = (NR > (n - jp)) ? (n - jp) : NR;

        for (int p = 0; p < k; p++)
        {
            for (int j = 0; j < real_NR; j++)
            {
                *(Bc++) = B[(jp + j) * csB + p * rsB];
            }

            // Not a multiple
            if (real_NR != NR)
            for (int j = real_NR; j < NR; j++)
            {
                *(Bc++) = 0.0;
            }
        }
    }
}
        

/* Pack a m x k panel of A into a MC x KC buffer.
   - The block is copied into Atilde a micro-panel at a time.
   - A is stored in row-major order.
   - The packed format is MC x KC, where MR-sized micro-panels are processed.
*/
void pack_panel_a(int m, int k, double *A, int rsA, int csA, double *Ac)
{
    for (int ip = 0; ip < m; ip += MR)
    {
        int real_MR = (MR > (m - ip)) ? (m - ip) : MR;

        for (int p = 0; p < k; p++)
        {
            for (int i = 0; i < real_MR; i++)
            {
                *(Ac++) = A[(ip + i) * rsA + p * csA]; 
            }

            for (int i = real_MR; i < MR; i++)
            {
                *(Ac++) = 0.0;
            }
        }
    }
}

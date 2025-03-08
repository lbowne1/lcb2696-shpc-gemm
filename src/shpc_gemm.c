#include "assignment3.h"
#include<immintrin.h>


void shpc_dgemm( int m, int n, int k,                                            
        double *A, int rsA, int csA,                                
        double *B, int rsB, int csB,                                
        double *C, int rsC, int csC )
{

    double *Bp;
    double *Ap;

    int main () 
    {
        // First loop partitions C and B into column panels.
        for ( int jc = 0; jc < n; jc += NC ) 
        {
            // Second loop partitions A and B over KC
            for ( int pc = 0; pc < k; pc += KC ) 
            {
                // Pack B
                for (int p = 0; p < KC; p++)     
                { 
                    for (int j = 0; j < NC; j++)   
                    { 
                        Bp[(pc * NC) + (p * NC) + j] = B[(pc + p) * rsB + (jc + j)];
                    }
                }

                // Third loop partitions C and A over MC
                for ( int ic = 0; ic < m; ic += MC ) 
                {

                    // Pack A
                    for (int p = 0; p < KC; p++) 
                    {   
                        for (int i = 0; i < MC; i++) 
                        {  
                            Ap[(ic * KC) + (p * MC) + i] = A[(pc + p) * rsA + (ic + i)];
                        }
                    }

                    // Fourth loop partitions Bp into column “slivers” of width nr.
                    for ( int jr = 0; jr < NC; jr += NR ) {

                        // Fifth loop partitions Ae into row slivers of height mr. 
                        for ( int ir = 0; ir < MC; ir += MR ) {
        
                            ukernel(Ap[ ir * MR ], rsA, csA,
                                    Bp[ jr * NR ], rsB, csB,
                                    C[ (ic + ir) * rsC + (jc + jr) * csC ], rsC, csC,
                                    KC);

                        }
                    }
                }
            }

        }

    }   

    /*
    BLIS micro-kernel
    then multiplies the current sliver of Ae by the current sliver
    of Be to update the corresponding mr × nr block of C. This
    micro-kernel performs a sequence of rank-1 updates (outer
    products) with columns from the sliver of Ae and rows from
    the sliver of B */
    void ukernel( double *A, int rsA, int csA,
                double *B, int rsB, int csB,
                double *C, int rsC, int csC,
                int k ) 
                {

        /* Declare vector registers to hold 4x4 C and load them */
        __m256d gamma_0123_0 = _mm256_loadu_pd( &C[0*rsC + 0*csC] );
        __m256d gamma_0123_1 = _mm256_loadu_pd( &C[0*rsC + 1*csC] );
        __m256d gamma_0123_2 = _mm256_loadu_pd( &C[0*rsC + 2*csC] );
        __m256d gamma_0123_3 = _mm256_loadu_pd( &C[0*rsC + 3*csC] );

        for ( int p=0; p<k; p++ )
        {   
        
            __m256d beta_p_0;
            /* Declare a vector register to hold the current column of A and load
            it with the four elements of that column. */
            __m256d alpha_0123_p = _mm256_loadu_pd( &A[ 0*rsA + p*csA ] );

            /* Load/broadcast beta( p,0 ). */
            beta_p_0 = _mm256_broadcast_sd( &B[ p*rsB + 0*csB ] );

            /* update the first column of C with the current column of A times
            beta ( p,0 ) */
            gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_0, gamma_0123_0 );


            /* Second Column */
            __m256d beta_p_1;
            __m256d alpha_0123_1 = _mm256_loadu_pd( &A[ 0*rsA + p*csA ] );
            beta_p_1 = _mm256_broadcast_sd( &B[ p*rsB + 0*csB ] );
            gamma_0123_1 = _mm256_fmadd_pd( alpha_0123_p, beta_p_1, gamma_0123_0 );


            /* Third Column */
            __m256d beta_p_2;
            __m256d alpha_0123_2 = _mm256_loadu_pd( &A[ 0*rsA + p*csA ] );
            beta_p_2 = _mm256_broadcast_sd( &B[ p*rsB + 0*csB ] );
            gamma_0123_2 = _mm256_fmadd_pd( alpha_0123_p, beta_p_2, gamma_0123_0 );


            /* Fourth Column */
            __m256d beta_p_3;
            __m256d alpha_0123_3 = _mm256_loadu_pd( &A[ 0*rsA + p*csA ] );
            beta_p_3 = _mm256_broadcast_sd( &B[ p*rsB + 0*csB ] );
            gamma_0123_3 = _mm256_fmadd_pd( alpha_0123_p, beta_p_3, gamma_0123_0 );
        }

        /* Store the updated results */
        _mm256_storeu_pd( &C[0*rsC + 0*csC], gamma_0123_0 );
        _mm256_storeu_pd( &C[0*rsC + 1*csC], gamma_0123_1 );
        _mm256_storeu_pd( &C[0*rsC + 2*csC], gamma_0123_2 );
        _mm256_storeu_pd( &C[0*rsC + 3*csC], gamma_0123_3 );    

        _mm256_storeu_pd( &C[4*rsC + 0*csC], gamma_4567_0 );
        _mm256_storeu_pd( &C[4*rsC + 1*csC], gamma_4567_1 );
        _mm256_storeu_pd( &C[4*rsC + 2*csC], gamma_4567_2 );
        _mm256_storeu_pd( &C[4*rsC + 3*csC], gamma_4567_3 ); 
    }
}
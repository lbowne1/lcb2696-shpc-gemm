#include "assignment3.h"
#include<immintrin.h>
#include <assert.h>

int MR = 4;
int NR = 4;

int KC = 48;
int MC = 48;
int NC = 48;


void shpc_dgemm( int m, int n, int k,                                            
        double *A, int rsA, int csA,                                
        double *B, int rsB, int csB,                                
        double *C, int rsC, int csC )
{


    //double *A_ptr = (double *) 
    // First loop partitions C and B into column panels.
    for ( int jc = 0; jc < n; jc += NC ) 
    {
        double *Bc = (double *) malloc(KC * NR * sizeof(double));
        double *Ac = (double *) malloc(MR * KC * sizeof(double));

        // Second loop partitions A and B over KC
        for ( int pc = 0; pc < k; pc += KC ) 
        {
            //Pack B
            for (int p = 0; p < KC; p++)     
            { 
                for (int j = 0; j < NR; j++)   
                { 
                    Bc[(p * NR) + j] = B[pc + (p * rsB) + (jc + j)];
                }
            }
            


            // Third loop partitions C and A over MC
            for ( int ic = 0; ic < m; ic += MC ) 
            {

                // Pack A
                for (int p = 0; p < KC; p++)
                {   
                    for (int i = 0; i < MR; i++) 
                    {  
                        Ac[(p * MR) + i] = A[(pc + (p * rsA)) + (ic + i)];
                    }
                }
                    */

                // Fourth loop partitions Bp into column “slivers” of width nr.
                for ( int jr = 0; jr < NC; jr += NR ) {

                    // Fifth loop partitions Ae into row slivers of height mr. 
                    for ( int ir = 0; ir < MC; ir += MR ) {
    
                        // if ((pc * csA) >= KC) assert(0);
                        // if (((ic * rsA) + (ir * rsA)) >= MC) assert(0);

                        // if ((pc * rsB) >= KC) assert(0);
                        // if (((jc + jr) * csB) >= NC) assert(0);

                        // if (((ic * rsC) + (ir * rsC)) >= MC) assert(0);
                        // if (((jc * csC) + (jr * csC)) >= NC) assert(0);
                        
                        // packed rsx and csx
                        microkernel(A + (pc * csA) + (ic + ir) * rsA, rsA, csA,
                        B + (jc + jr) * csB + (pc * rsB), rsB, csB,
                        C + (ic + ir) * rsC + (jc + jr) * csC, rsC, csC,
                        KC);
                    }
                }
            }
        }
        //free(Bc);
        //free(Ac);
    }

}   

/*
BLIS micro-kernel
then multiplies the current sliver of Ae by the current sliver
of Be to update the corresponding mr × nr block of C. This
micro-kernel performs a sequence of rank-1 updates (outer
products) with columns from the sliver of Ae and rows from
the sliver of B */
void microkernel( double *A, int rsA, int csA,
            double *B, int rsB, int csB,
            double *C, int rsC, int csC,
            int k ) 
{
    {
        __m256d gamma_0123_0 = _mm256_loadu_pd( &C[0*rsC + 0*csC] );
        __m256d gamma_0123_1 = _mm256_loadu_pd( &C[0*rsC + 1*csC] );
        __m256d gamma_0123_2 = _mm256_loadu_pd( &C[0*rsC + 2*csC] );
        __m256d gamma_0123_3 = _mm256_loadu_pd( &C[0*rsC + 3*csC] );

        for ( int p=0; p<k; p++ )
        {
            //for ( int j=0; j<NR; j++) 
            //{
                /* Declare vector registers to hold element j of C and hold it */
                //gamma_0123_j = _mm256_loadu_pd( &C[0*rsC + j*csC] );

                /* Declare vector register for load/broadcasting beta( p,j ) */
                __m256d beta_p_0;
                __m256d beta_p_1;
                __m256d beta_p_2;
                __m256d beta_p_3;
        
                /* Declare a vector register to hold the current column of A and load
                it with the four elements of that column. */
                __m256d alpha_0123_p = _mm256_loadu_pd( &A[ 0*rsA + p*csA ] );
        
                /* Load/broadcast beta( p,0 ). */
                beta_p_0 = _mm256_broadcast_sd( &B[ p*rsB + 0*csB ] );
                beta_p_1 = _mm256_broadcast_sd( &B[ p*rsB + 1*csB ] );
                beta_p_2 = _mm256_broadcast_sd( &B[ p*rsB + 2*csB ] );
                beta_p_3 = _mm256_broadcast_sd( &B[ p*rsB + 3*csB ] );
        
                /* update the first column of C with the current column of A times
                beta ( p,0 ) */
                gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_0, gamma_0123_0 );
                gamma_0123_1 = _mm256_fmadd_pd( alpha_0123_p, beta_p_1, gamma_0123_1 );
                gamma_0123_2 = _mm256_fmadd_pd( alpha_0123_p, beta_p_2, gamma_0123_2 );
                gamma_0123_3 = _mm256_fmadd_pd( alpha_0123_p, beta_p_3, gamma_0123_3 );

                /* Store the updated results */
                //_mm256_storeu_pd( &C[0*rsC + j*csC], gamma_0123_j );

            //}
        } 
        _mm256_storeu_pd( &C[0*rsC + 0*csC], gamma_0123_0 );
        _mm256_storeu_pd( &C[0*rsC + 1*csC], gamma_0123_1 );
        _mm256_storeu_pd( &C[0*rsC + 2*csC], gamma_0123_2 );
        _mm256_storeu_pd( &C[0*rsC + 3*csC], gamma_0123_3 );


    }
}
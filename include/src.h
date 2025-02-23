
/* Computes C = AB + C */
void shpc_dgemm( int m, int n, int k, 
                    double *A, int rsA, int csA, 
                    double *B, int rsB, int csB, 
                    double *C, int rsC, int csC );
                

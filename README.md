# High Performance GEMM 
Implementation of a high-performance GEMM optimized for UTCS machine onyx.
Block Sizes:
MR = 8;
NR = 6;

KC = 256;
MC = 968;
NC = 11262;

I parallelized the second loop around the microokernel as it allowed me the 
best performance and lessened overhead by preventing having to allocate
multiple blocks of A.
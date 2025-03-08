#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>


#define dabs( x ) ( (x) < 0 ? -(x) : x )
#define max(x, y) (((x) > (y)) ? (x) : (y))

#define int MR = 4;
#define int NR = 4;

#define int KC = 240;
#define int MC = 240;
#define int NC = 240;


#include "blis.h"

#include "src.h"
#include "test.h"                                                        
#include "util.h"

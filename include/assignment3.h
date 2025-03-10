#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <assert.h>


#define dabs( x ) ( (x) < 0 ? -(x) : x )
#define max(x, y) (((x) > (y)) ? (x) : (y))

extern int MR;
extern int NR;
extern int KC;
extern int MC;
extern int NC;

#include "blis.h"

#include "src.h"
#include "test.h"                                                        
#include "util.h"

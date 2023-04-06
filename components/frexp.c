#include <stdio.h>
#include <math.h>

int main () {
   double x = (0.6599291563034058 / (2 * 2.7218637466430664 )  / pow(2,20)), fraction;
   int e;
   
   fraction = frexp(x, &e);
   printf("x = float(%.6lf) = int32(%.2lf) * 2^%d\n", x, fraction * (1ll << 31), e);
   
   return(0);
}
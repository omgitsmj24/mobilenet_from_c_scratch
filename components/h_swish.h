#include "stdio.h"
#include "math.h"

float h_swish(float x) {
  return x * fminf(fmaxf(x + 3, 0.0f), 6.0f) / 6.0f;
}
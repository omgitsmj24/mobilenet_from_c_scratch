#include "stdio.h"
#include "math.h"

void relu(float *input, float *output, int size) {
  for (int i = 0; i < size; i++) {
    output[i] = fmaxf(input[i], 0.0f);
  }
}
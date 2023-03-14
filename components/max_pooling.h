#include "stdio.h"
#include "math.h"
#include "float.h"

void max_pool(float *input, float *output, int input_h, int input_w,
              int pool_h, int pool_w) {

  // Compute output size
  int output_h = input_h / pool_h;
  int output_w = input_w / pool_w;

  // Perform max pooling
  for (int i = 0; i < output_h; i++) {
    for (int j = 0; j < output_w; j++) {
      int output_idx = i * output_w + j;
      output[output_idx] = -FLT_MAX;

      for (int k = 0; k < pool_h; k++) {
        for (int l = 0; l < pool_w; l++) {
          int input_idx = (i * pool_h + k) * input_w + (j * pool_w + l);
          output[output_idx] = fmaxf(output[output_idx], input[input_idx]);
        }
      }
    }
  }
}
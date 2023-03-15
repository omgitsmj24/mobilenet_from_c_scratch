#include "stdio.h"
#include "math.h"
#include "float.h"

void avg_pool(float *input, float *output, int input_h, int input_w,
              int pool_h, int pool_w) {

  // Compute output size
  int output_h = input_h / pool_h;
  int output_w = input_w / pool_w;

  // Perform average pooling
  for (int i = 0; i < output_h; i++) {
    for (int j = 0; j < output_w; j++) {
      int output_idx = i * output_w + j;
      output[output_idx] = 0.0f;

      for (int k = 0; k < pool_h; k++) {
        for (int l = 0; l < pool_w; l++) {
          int input_idx = (i * pool_h + k) * input_w + (j * pool_w + l);
          output[output_idx] += input[input_idx];
        }
      }

      output[output_idx] /= (float)(pool_h * pool_w);
    }
  }
}
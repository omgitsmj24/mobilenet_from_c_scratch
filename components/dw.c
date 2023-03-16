#include "dw.h"
#include <stdio.h>

void depthwise_conv(float *input, float *output, float *kernel, int input_h,
                    int input_w, int kernel_h, int kernel_w) {

  // Compute output size
  int output_h = input_h - kernel_h + 1;
  int output_w = input_w - kernel_w + 1;

  // Perform depthwise convolution
  for (int i = 0; i < output_h; i++) {
    for (int j = 0; j < output_w; j++) {
      for (int k = 0; k < kernel_h; k++) {
        for (int l = 0; l < kernel_w; l++) {
          int input_idx = (i + k) * input_w + (j + l);
          int kernel_idx = k * kernel_w + l;
          int output_idx = i * output_w + j;

          output[output_idx] += input[input_idx] * kernel[kernel_idx];
        }
      }
    }
  }
}
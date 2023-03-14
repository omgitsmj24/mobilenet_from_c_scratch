#include "stdio.h"
#include "math.h"

void batch_norm(float *input, float *output, float *mean, float *variance,
                float *gamma, float *beta, int size, float eps) {

  // Compute mean
  for (int i = 0; i < size; i++) {
    mean[i] = 0.0f;
    for (int j = 0; j < size; j++) {
      mean[i] += input[i * size + j];
    }
    mean[i] /= size;
  }

  // Compute variance
  for (int i = 0; i < size; i++) {
    variance[i] = 0.0f;
    for (int j = 0; j < size; j++) {
      variance[i] += powf(input[i * size + j] - mean[i], 2);
    }
    variance[i] /= size;
    variance[i] += eps;
  }

  // Compute normalized output
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      output[i * size + j] = gamma[i] * (input[i * size + j] - mean[i]) / sqrtf(variance[i]);
      output[i * size + j] += beta[i];
    }
  }
}

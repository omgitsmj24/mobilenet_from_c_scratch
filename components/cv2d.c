#include <cv2d.h>
#include <stdio.h>
#include <stdlib.h>

void conv2d(float ***input, int input_size, int input_channels, float ****kernel, int kernel_size, int num_kernels, int kernel_depth, 
            float ***output, float *bias, int stride, int padding){
    int output_size = (input_size - kernel_size + 2*padding)/stride + 1;

    for (int k = 0; k < num_kernels; k++) {
        for (int l = 0; l < kernel_depth; l++) {
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < output_size; j++) {
                    for (int m = 0; m < kernel_size; m++) {
                        for (int n = 0; n < kernel_size; n++) {
                            output[i][j][k] += input[i*stride + m][j*stride + n][l] * kernel[m][n][l][k];
                        }
                    }
                    output[i][j][k] += bias[k];
                }
            }
        }
    }
}
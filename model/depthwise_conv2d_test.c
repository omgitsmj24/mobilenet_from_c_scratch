#include "stdio.h"
#include "stdlib.h"

#define INPUT_WIDTH  224
#define INPUT_HEIGHT 224
#define INPUT_CHANNEL 3
#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 1
#define OUTPUT_CHANNEL 16

float input[INPUT_WIDTH][INPUT_HEIGHT][INPUT_CHANNEL];
float depthwise_kernel[KERNEL_SIZE][KERNEL_SIZE][INPUT_CHANNEL];
float pointwise_kernel[INPUT_CHANNEL][OUTPUT_CHANNEL];
float output[INPUT_WIDTH][INPUT_HEIGHT][OUTPUT_CHANNEL];

int main(int argc, char** argv) {
    // Read input image (in this example, we assume the input image is already in float format)
    // Fill the input with some random data as an example

    for (int i = 0; i < INPUT_WIDTH; i++) {
        for (int j = 0; j < INPUT_HEIGHT; j++) {
            for (int k = 0; k < INPUT_CHANNEL; k++) {
                input[i][j][k] = (float)rand() / (float)RAND_MAX;
            }
        }
    }
    // Initialize depthwise kernel with some random data as an example
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            for (int k = 0; k < INPUT_CHANNEL; k++) {
                depthwise_kernel[i][j][k] = (float)rand() / (float)RAND_MAX;
            }
        }
    }
    // Initialize pointwise kernel with some random data as an example
    for (int i = 0; i < INPUT_CHANNEL; i++) {
        for (int j = 0; j < OUTPUT_CHANNEL; j++) {
            pointwise_kernel[i][j] = (float) rand() / (float)RAND_MAX;
        }
    }
    // Apply depthwise separable convolution
    for (int oc = 0; oc < OUTPUT_CHANNEL; oc++) {
        for (int oy = 0; oy < INPUT_HEIGHT; oy++) {
            for (int ox = 0; ox < INPUT_WIDTH; ox++) {
                float sum = 0;
                for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        for (int ic = 0; ic < INPUT_CHANNEL; ic++) {
                            int ix = ox + kx - PADDING;
                            int iy = oy + ky - PADDING;
                            if (ix >= 0 && ix < INPUT_WIDTH && iy >= 0 && iy < INPUT_HEIGHT) {
                                sum += input[ix][iy][ic] * depthwise_kernel[kx][ky][ic];
                            }
                        }
                    }
                }
                output[ox][oy][oc] = sum;
            }
        }
    }

    // Apply pointwise convolution
    for (int oc = 0; oc < OUTPUT_CHANNEL; oc++) {
        for (int oy = 0; oy < INPUT_HEIGHT; oy++) {
            for (int ox = 0; ox < INPUT_WIDTH; ox++) {
                float sum = 0;
                for (int ic = 0; ic < INPUT_CHANNEL; ic++) {
                    sum += output[ox][oy][ic] * pointwise_kernel[ic][oc];
                }
                output[ox][oy][oc] = sum;
            }
        }
    }

    // Print output for debugging
    for (int oc = 0; oc < OUTPUT_CHANNEL; oc++) {
        for (int oy = 0; oy < INPUT_HEIGHT; oy++) {
            for (int ox = 0; ox < INPUT_WIDTH; ox++) {
                printf("%f ", output[ox][oy][oc]);
            }
            printf(" ");
        }

    }
    return 0;
}

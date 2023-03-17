#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define INPUT_SIZE 5
#define INPUT_DEPTH 3
#define KERNEL_SIZE 2
#define NUM_FILTERS 2
#define STRIDE 1

int main() {
    // Define input and kernel arrays
    float input[INPUT_SIZE][INPUT_SIZE][INPUT_DEPTH];
    float kernel[KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][NUM_FILTERS];

    // Initialize input and kernel arrays with random values
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            for (int k = 0; k < INPUT_DEPTH; k++) {
                input[i][j][k] = (float)rand()/(float)(RAND_MAX/10.0);
                for (int l = 0; l < NUM_FILTERS; l++) {
                    kernel[i][j][k][l] = (float)rand()/(float)(RAND_MAX/10.0);
                }
            }
        }
    }

    // Perform depthwise convolution
    int output_size = (INPUT_SIZE - KERNEL_SIZE) / STRIDE + 1;
    float output[output_size][output_size][INPUT_DEPTH * NUM_FILTERS];
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            for (int k = 0; k < INPUT_DEPTH; k++) {
                for (int l = 0; l < NUM_FILTERS; l++) {
                    float sum = 0.0;
                    for (int m = 0; m < KERNEL_SIZE; m++) {
                        for (int n = 0; n < KERNEL_SIZE; n++) {
                            int x = i * STRIDE + m;
                            int y = j * STRIDE + n;
                            sum += input[x][y][k] * kernel[m][n][k][l];
                        }
                    }
                    output[i][j][k * NUM_FILTERS + l] = sum;
                }
            }
        }
    }

    // Print output
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            for (int k = 0; k < INPUT_DEPTH * NUM_FILTERS; k++) {
                printf("%.2f ", output[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}

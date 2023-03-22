#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 28
#define KERNEL_SIZE 3
#define STRIDE 1
#define NUM_FILTERS 8
#define OUTPUT_SIZE ((INPUT_SIZE - KERNEL_SIZE) / STRIDE + 1)

// Define input and output tensors
float input[INPUT_SIZE][INPUT_SIZE];
float output[OUTPUT_SIZE][OUTPUT_SIZE][NUM_FILTERS];

// Define convolution kernel tensor
float kernel[KERNEL_SIZE][KERNEL_SIZE][NUM_FILTERS];

// Initialize input tensor with random values
void init_input() {
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            input[i][j] = rand() / (float)RAND_MAX;
        }
    }
}

// Initialize kernel tensor with random values
void init_kernel() {
    for (int k = 0; k < NUM_FILTERS; k++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                kernel[i][j][k] = rand() / (float)RAND_MAX;
            }
        }
    }
}

// Perform convolution operation
void conv2d() {
    for (int k = 0; k < NUM_FILTERS; k++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                float sum = 0.0;
                for (int x = 0; x < KERNEL_SIZE; x++) {
                    for (int y = 0; y < KERNEL_SIZE; y++) {
                        int row = i * STRIDE + x;
                        int col = j * STRIDE + y;
                        sum += input[row][col] * kernel[x][y][k];
                    }
                }
                output[i][j][k] = sum;
            }
        }
    }
}

// Print output tensor
void print_output() {
    for (int k = 0; k < NUM_FILTERS; k++) {
        printf("Filter %d:\n", k);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                printf("%f ", output[i][j][k]);
            }
            printf("\n");
        }
    }
}

int main() {
    // Initialize input and kernel tensors
    init_input();
    init_kernel();

    // Perform convolution operation
    conv2d();

    // Print output tensor
    print_output();

    return 0;
}

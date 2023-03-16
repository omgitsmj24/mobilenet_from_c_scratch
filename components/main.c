// #include "dw.h"
// #include "input.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//INPUT
#define INPUT_WIDTH 4
#define INPUT_HEIGHT 4
#define INPUT_CHANNELS 3


//DEPTHWISE CONVOLUTION
#define DEPTHWISE_FILTER_SIZE 3
#define DEPTHWISE_FILTER_DEPTH 8
#define DEPTHWISE_STRIDE 1
#define DEPTHWISE_PADDING 0

//OUTPUT1
#define OUTPUT1_SIZE ((INPUT_WIDTH - DEPTHWISE_FILTER_SIZE) / DEPTHWISE_STRIDE + 1)

//POINTWISE CONVOLUTION
#define POINTWISE_FILTER_SIZE 1
#define POINTWISE_FILTER_DEPTH 3
#define POINTWISE_STRIDE 1
#define POINTWISE_PADDING 0

//OUTPUT2
#define OUTPUT2_SIZE ((OUTPUT1_SIZE - POINTWISE_FILTER_SIZE) / POINTWISE_STRIDE + 1)

// Define input and output tensors
float input[INPUT_WIDTH][INPUT_HEIGHT];
float output1[OUTPUT1_SIZE][OUTPUT1_SIZE][DEPTHWISE_FILTER_DEPTH];
float output2[OUTPUT2_SIZE][OUTPUT2_SIZE][POINTWISE_FILTER_DEPTH];

// Define depthwise convolution kernel tensor
float depthwise_kernel[DEPTHWISE_FILTER_SIZE][DEPTHWISE_FILTER_SIZE][DEPTHWISE_FILTER_DEPTH];

// Define pointwise convolution kernel tensor
float pointwise_kernel[POINTWISE_FILTER_SIZE][POINTWISE_FILTER_SIZE][POINTWISE_FILTER_DEPTH];

// Init input tensor with random valuesS
void init_input() {
    for (int i = 0; i < INPUT_WIDTH; i++) {
        for (int j = 0; j < INPUT_WIDTH; j++) {
            input[i][j] = rand() / (float)RAND_MAX;
        }
    }
}

// Initialize depthwise kernel tensor with random values
void init_depthwise_kernel() {
    for (int k = 0; k < DEPTHWISE_FILTER_DEPTH; k++) {
        for (int i = 0; i < DEPTHWISE_FILTER_SIZE; i++) {
            for (int j = 0; j < DEPTHWISE_FILTER_SIZE; j++) {
                depthwise_kernel[i][j][k] = rand() / (float)RAND_MAX;
            }
        }
    }
}

// Initialize pointwise kernel tensor with random values
void init_pointwise_kernel() {
    for (int k = 0; k < POINTWISE_FILTER_DEPTH; k++) {
        for (int i = 0; i < POINTWISE_FILTER_SIZE; i++) {
            for (int j = 0; j < POINTWISE_FILTER_SIZE; j++) {
                pointwise_kernel[i][j][k] = rand() / (float)RAND_MAX;
            }
        }
    }
}

//Perform depthwise convolution
void depthwise_conv2d() {
    for (int k = 0; k < DEPTHWISE_FILTER_DEPTH; k++) {
        for (int i = 0; i < OUTPUT1_SIZE; i++) {
            for (int j = 0; j < OUTPUT1_SIZE; j++) {
                float sum = 0;
                for (int m = 0; m < DEPTHWISE_FILTER_SIZE; m++) {
                    for (int n = 0; n < DEPTHWISE_FILTER_SIZE; n++) {
                        sum += input[i + m][j + n] * depthwise_kernel[m][n][k];
                    }
                }
                output1[i][j][k] = sum;
            }
        }
    }
}

void pointwise_conv2d(){
    for (int k = 0; k < POINTWISE_FILTER_DEPTH; k++) {
        for (int i = 0; i < OUTPUT2_SIZE; i++) {
            for (int j = 0; j < OUTPUT2_SIZE; j++) {
                float sum = 0;
                for (int m = 0; m < POINTWISE_FILTER_SIZE; m++) {
                    for (int n = 0; n < POINTWISE_FILTER_SIZE; n++) {
                        sum += output1[i + m][j + n][k] * pointwise_kernel[m][n][k];
                    }
                }
                output2[i][j][k] = sum;
            }
        }
    }
}

// Print output1 tensor
void print_output1() {
    for (int k = 0; k < DEPTHWISE_FILTER_DEPTH; k++) {
        printf("Filter %d:\n", k);
        for (int i = 0; i < OUTPUT1_SIZE; i++) {
            for (int j = 0; j < OUTPUT1_SIZE; j++) {
                printf("%f ", output1[i][j][k]);
            }
            printf("\n");
        }
    }
}

// Print output2 tensor
void print_output2() {
    for (int k = 0; k < POINTWISE_FILTER_DEPTH; k++) {
        printf("Filter %d:\n", k);
        for (int i = 0; i < OUTPUT2_SIZE; i++) {
            for (int j = 0; j < OUTPUT2_SIZE; j++) {
                printf("%f ", output2[i][j][k]);
            }
            printf("\n");
        }
    }
}

int main() {
    
    // Initialize input 
    init_input();
    // Initialize depthwise kernel tensor with random values
    init_depthwise_kernel();
    // Initialize pointwise kernel tensor with random values
    init_pointwise_kernel();
    // Perform depthwise convolution
    depthwise_conv2d();
    // Print output1 tensor
    print_output1();
    // Perform pointwise convolution
    pointwise_conv2d();
    // Print output2 tensor
    print_output2();
    return 0;
    
}
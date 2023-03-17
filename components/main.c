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
#define DEPTHWISE_FILTER_SIZE 2
#define DEPTHWISE_FILTER_DEPTH INPUT_CHANNELS
#define DEPTHWISE_STRIDE 1
#define DEPTHWISE_PADDING 0
#define DEPTHWISE_NUM_FILTERS 3

//OUTPUT1
#define OUTPUT1_SIZE ((INPUT_WIDTH - DEPTHWISE_FILTER_SIZE) / DEPTHWISE_STRIDE + 1)

//POINTWISE CONVOLUTION
#define POINTWISE_FILTER_SIZE 1
#define POINTWISE_FILTER_DEPTH 3
#define POINTWISE_STRIDE 1
#define POINTWISE_PADDING 0
#define POINTWISE_NUM_FILTERS 3

//OUTPUT2
#define OUTPUT2_SIZE ((OUTPUT1_SIZE - POINTWISE_FILTER_SIZE) / POINTWISE_STRIDE + 1)

// Define input and output tensors
float input[INPUT_WIDTH][INPUT_HEIGHT][INPUT_CHANNELS];
float output1[OUTPUT1_SIZE][OUTPUT1_SIZE][DEPTHWISE_FILTER_DEPTH][DEPTHWISE_NUM_FILTERS];
float output2[OUTPUT2_SIZE][OUTPUT2_SIZE][POINTWISE_FILTER_DEPTH][POINTWISE_NUM_FILTERS];

// Define depthwise convolution kernel tensor
float depthwise_kernel[DEPTHWISE_FILTER_SIZE][DEPTHWISE_FILTER_SIZE][DEPTHWISE_FILTER_DEPTH][DEPTHWISE_NUM_FILTERS];

// Define pointwise convolution kernel tensor
float pointwise_kernel[POINTWISE_FILTER_SIZE][POINTWISE_FILTER_SIZE][POINTWISE_FILTER_DEPTH][POINTWISE_NUM_FILTERS];

// Init input tensor with random values
void init_input() {
    printf("Input: \n");
    for (int k = 0; k < INPUT_CHANNELS; k++){
        for (int i = 0; i < INPUT_WIDTH; i++) {
            for (int j = 0; j < INPUT_WIDTH; j++) {
                input[i][j][k] = rand() / (float)RAND_MAX;
                printf("%f ", input[i][j][k]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

// Initialize depthwise kernel tensor with random values
void init_depthwise_kernel() {
    for (int l = 0; l < DEPTHWISE_NUM_FILTERS; l++) {
        for (int k = 0; k < DEPTHWISE_FILTER_DEPTH; k++) {
            for (int i = 0; i < DEPTHWISE_FILTER_SIZE; i++) {
                for (int j = 0; j < DEPTHWISE_FILTER_SIZE; j++) {
                    depthwise_kernel[i][j][k][l] = rand() / (float)RAND_MAX;
                }
            }
        }
    }
}

// Initialize pointwise kernel tensor with random values
void init_pointwise_kernel() {
    for (int l = 0; l < POINTWISE_NUM_FILTERS; l++) {
        for (int k = 0; k < POINTWISE_FILTER_DEPTH; k++) {
            for (int i = 0; i < POINTWISE_FILTER_SIZE; i++) {
                for (int j = 0; j < POINTWISE_FILTER_SIZE; j++) {
                    pointwise_kernel[i][j][k][l] = rand() / (float)RAND_MAX;
                }
            }
        }
    }
}

//Perform depthwise convolution
void depthwise_conv2d() {
    for (int l = 0; l < DEPTHWISE_NUM_FILTERS; l++){
        for (int k = 0; k < DEPTHWISE_FILTER_DEPTH; k++) {
            for (int i = 0; i < OUTPUT1_SIZE; i++) {
                for (int j = 0; j < OUTPUT1_SIZE; j++) {
                    float sum = 0;
                    for (int m = 0; m < DEPTHWISE_FILTER_SIZE; m++) {
                        for (int n = 0; n < DEPTHWISE_FILTER_SIZE; n++) {
                            int x = i * DEPTHWISE_STRIDE + m;
                            int y = j * DEPTHWISE_STRIDE + n;
                            sum += input[x][y][k] * depthwise_kernel[m][n][k][l];
                        }
                    }
                    output1[i][j][k][l] = sum;
                }
            }
        }
    }
}

void pointwise_conv2d(){
    for (int l = 0; l < POINTWISE_NUM_FILTERS; l++) {
        for (int k = 0; k < POINTWISE_FILTER_DEPTH; k++) {
            for (int o = 0; o < INPUT_CHANNELS; o)
                for (int i = 0; i < OUTPUT2_SIZE; i++) {
                    for (int j = 0; j < OUTPUT2_SIZE; j++) {
                        float sum = 0;
                        for (int m = 0; m < POINTWISE_FILTER_SIZE; m++) {
                            for (int n = 0; n < POINTWISE_FILTER_SIZE; n++) {
                                sum += output1[i * POINTWISE_STRIDE + m][j * POINTWISE_STRIDE + n][k][l] * pointwise_kernel[m][n][k][l];
                            }
                        }
                        output2[i][j][k][l] = sum;
                    }
                }
        }
    }
}

// Print output1 tensor
void print_output1() {
    printf("\nDEPTHWISE OUPUT: \n");
    for (int l = 0; l < DEPTHWISE_FILTER_SIZE; l++) {
        printf("FILTER %d:\n", l);
        for (int k = 0; k < DEPTHWISE_FILTER_DEPTH; k++) {
            printf("DEPTH %d:\n", k);
            for (int i = 0; i < OUTPUT1_SIZE; i++) {
                for (int j = 0; j < OUTPUT1_SIZE; j++) {
                    printf("%f ", output1[i][j][k][l]);
                }
                printf("\n");
            }
        }
    }
}

// Print output2 tensor
void print_output2() {
    printf("\nPOINTWISE OUTPUT: \n");
    for (int l = 0; l < POINTWISE_FILTER_SIZE; l++) {
        for (int k = 0; k < POINTWISE_FILTER_DEPTH; k++) {
            printf("Filter %d:\n", k);
            for (int i = 0; i < OUTPUT2_SIZE; i++) {
                for (int j = 0; j < OUTPUT2_SIZE; j++) {
                    printf("%f ", output2[i][j][k][l]);
                }
                printf("\n");
            }
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
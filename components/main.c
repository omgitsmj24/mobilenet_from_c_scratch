// #include "dw.h"
// #include "input.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//INPUT
#define INPUT_SIZE 5
#define INPUT_CHANNELS 3

//CONV2D 
#define CONV2D_FILTER_SIZE 3
#define CONV2D_FILTER_DEPTH 3
#define CONV2D_STRIDE 2
#define CONV2D_PADDING 1
#define CONV2D_NUM_FILTERS 3

//OUTPUT
#define OUTPUT_SIZE ((INPUT_SIZE - CONV2D_FILTER_SIZE) / CONV2D_STRIDE + 1)

//DEPTHWISE CONVOLUTION
#define DEPTHWISE_FILTER_SIZE 3
#define DEPTHWISE_FILTER_DEPTH INPUT_CHANNELS
#define DEPTHWISE_STRIDE 1
#define PADDING 1
#define DEPTHWISE_NUM_FILTERS 1

//PADDED INPUT
#define PADDED_INPUT_SIZE (INPUT_SIZE + 2 * PADDING)

//OUTPUT1
#define OUTPUT1_SIZE ((PADDED_INPUT_SIZE - DEPTHWISE_FILTER_SIZE) / DEPTHWISE_STRIDE + 1)

//POINTWISE CONVOLUTION
#define POINTWISE_FILTER_SIZE 1
#define POINTWISE_FILTER_DEPTH 3
#define POINTWISE_STRIDE 1
#define POINTWISE_NUM_FILTERS 8

//OUTPUT2
#define OUTPUT2_SIZE ((OUTPUT1_SIZE - POINTWISE_FILTER_SIZE) / POINTWISE_STRIDE + 1)

// Define input and output tensors
int input[INPUT_SIZE][INPUT_SIZE][INPUT_CHANNELS];
int output[OUTPUT_SIZE][OUTPUT_SIZE][CONV2D_FILTER_DEPTH][CONV2D_NUM_FILTERS];
int output1[OUTPUT1_SIZE][OUTPUT1_SIZE][DEPTHWISE_FILTER_DEPTH];
int output2[OUTPUT2_SIZE][OUTPUT2_SIZE][POINTWISE_FILTER_DEPTH][POINTWISE_NUM_FILTERS];
int padded_input[PADDED_INPUT_SIZE][PADDED_INPUT_SIZE][INPUT_CHANNELS];

// Define convolution kernel tensor
int conv2d_kernel[CONV2D_FILTER_SIZE][CONV2D_FILTER_SIZE][CONV2D_FILTER_DEPTH][CONV2D_NUM_FILTERS];

// Define depthwise convolution kernel tensor
int depthwise_kernel[DEPTHWISE_FILTER_SIZE][DEPTHWISE_FILTER_SIZE][DEPTHWISE_FILTER_DEPTH];

// Define pointwise convolution kernel tensor
int pointwise_kernel[POINTWISE_FILTER_SIZE][POINTWISE_FILTER_SIZE][POINTWISE_FILTER_DEPTH][POINTWISE_NUM_FILTERS];


// Init padded input tensor with random values
void init_input() {
    // Initialize input tensor with random values
    for (int d = 0; d < INPUT_CHANNELS; d++) {
        for (int i = 0 ; i < INPUT_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                input[i][j][d] = rand() % 256;
            }
        }
    }

    // Initialize padded input tensor with zeros
    for (int d = 0; d < INPUT_CHANNELS; d++) {
        for (int i = 0 ; i < PADDED_INPUT_SIZE; i++) {
            for (int j = 0; j < PADDED_INPUT_SIZE; j++) {
                padded_input[i][j][d] = 0;
            }
        }
    }

    // Copy input tensor into padded input tensor
    for (int d = 0; d < INPUT_CHANNELS; d++) {
        for (int i = 0 ; i < INPUT_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                padded_input[i + PADDING][j + PADDING][d] = input[i][j][d];
            }
        }
    }

    // Print padded input
    printf("Padded input: \n");
    for (int d = 0; d < INPUT_CHANNELS; d++) {
        printf("Channel %d: \n", d);
        for (int i = 0; i < PADDED_INPUT_SIZE; i++) {
            for (int j = 0; j < PADDED_INPUT_SIZE; j++) {
                printf("%d ", padded_input[i][j][d]);
            }
            printf("\n");
    }
    printf("\n");
    }
}

void init_conv2d_kernel() {
    for (int l = 0; l < CONV2D_NUM_FILTERS; l++) {
        for (int k = 0; k < CONV2D_FILTER_DEPTH; k++) {
            for (int i = 0; i < CONV2D_FILTER_SIZE; i++) {
                for (int j = 0; j < CONV2D_FILTER_SIZE; j++) {
                    conv2d_kernel[i][j][k][l] = rand() % 5 - 2;
                }
            }
        }
    }
}

// Initialize depthwise kernel tensor with random values
void init_depthwise_kernel() {
    for (int k = 0; k < DEPTHWISE_FILTER_DEPTH; k++) {
        for (int i = 0; i < DEPTHWISE_FILTER_SIZE; i++) {
            for (int j = 0; j < DEPTHWISE_FILTER_SIZE; j++) {
                depthwise_kernel[i][j][k] = rand() % 5 - 2;
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
                    pointwise_kernel[i][j][k][l] = rand() % 5 - 2;
                }
            }
        }
    }
}

//Perform conv2d 

void conv2d() {
    for (int k = 0; k < CONV2D_NUM_FILTERS; k++) {
        for (int l = 0; l < CONV2D_FILTER_DEPTH; l++) {
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    int sum = 0;
                    for (int m = 0; m < CONV2D_FILTER_SIZE; m++) {
                        for (int n = 0; n < CONV2D_FILTER_SIZE; n++) {
                            int x = i * CONV2D_STRIDE + m;
                            int y = j * CONV2D_STRIDE + n;
                            sum += padded_input[x][y][l] * conv2d_kernel[m][n][l][k];
                        }
                    }
                    output[i][j][l][k] = sum;
                }
            }
        }
    }
}

//Perform depthwise convolution
void depthwise_conv2d() {
    for (int k = 0; k < DEPTHWISE_FILTER_DEPTH; k++) {
        for (int i = 0; i < OUTPUT1_SIZE; i++) {
            for (int j = 0; j < OUTPUT1_SIZE; j++) {
                int sum = 0;
                for (int m = 0; m < DEPTHWISE_FILTER_SIZE; m++) {
                    for (int n = 0; n < DEPTHWISE_FILTER_SIZE; n++) {
                        int x = i * DEPTHWISE_STRIDE + m;
                        int y = j * DEPTHWISE_STRIDE + n;
                        sum += output[x][y][k][l] * depthwise_kernel[m][n][k];
                    }
                }
                output1[i][j][k] = sum;
            }
        }
    }
}

void pointwise_conv2d () {
    for (int l = 0; l < POINTWISE_NUM_FILTERS; l++) {
        for (int k = 0; k < POINTWISE_FILTER_DEPTH; k++) {
            for (int i = 0; i < OUTPUT2_SIZE; i++) {
                for (int j = 0; j < OUTPUT2_SIZE; j++) {
                    int sum = 0;
                    int x = i * POINTWISE_STRIDE;
                    int y = j * POINTWISE_STRIDE;
                    sum += output1[x][y][k] * pointwise_kernel[0][0][k][l];
                    output2[i][j][k][l] = sum;
                }
            }
        }
    }
}

void print_conv2d_kernel() {
    printf("\nConv2d Kernel: \n");
    for (int k = 0; k < CONV2D_NUM_FILTERS; k++) {
        printf("Filter %d:\n", k);
        for (int l = 0; l < CONV2D_FILTER_DEPTH; l++) {
            printf("Depth %d:\n", l);
            for (int i = 0; i < CONV2D_FILTER_SIZE; i++) {
                for (int j = 0; j < CONV2D_FILTER_SIZE; j++) {
                    printf("%d ", conv2d_kernel[i][j][l][k]);
                }
                printf("\n");
            }
        }
    }
}

void print_depthwise_kernel() {
    printf("\nDepthwise Kernel: \n");
    for (int k = 0; k < DEPTHWISE_FILTER_DEPTH; k++) {
        printf("Depth %d:\n", k);
        for (int i = 0; i < DEPTHWISE_FILTER_SIZE; i++) {
            for (int j = 0; j < DEPTHWISE_FILTER_SIZE; j++) {
                printf("%d ", depthwise_kernel[i][j][k]);
            }
            printf("\n");
        }
    }
}

void print_pointwise_kernel() {
    printf("\nPointwise Kernel: \n");
    for (int k = 0; k < POINTWISE_NUM_FILTERS; k++) {
        printf("Filter %d:\n", k);
        for (int l = 0; l < POINTWISE_FILTER_DEPTH; l++) {
            printf("Depth %d:\n", l);
            for (int i = 0; i < POINTWISE_FILTER_SIZE; i++) {
                for (int j = 0; j < POINTWISE_FILTER_SIZE; j++) {
                    printf("%d ", pointwise_kernel[i][j][l][k]);
                }
                printf("\n");
            }
        }
    }
}

// Print output tensor
void print_output() {
    printf("\nOUTPUT: \n");
    for (int k = 0; k < CONV2D_NUM_FILTERS; k++) {
        printf("Filter %d:\n", k);
        for (int l = 0; l < CONV2D_FILTER_DEPTH; l++) {
            printf("Depth %d:\n", l);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    printf("%d ", output[i][j][l][k]);
                }
                printf("\n");
            }
        }
    }
}

// Print output1 tensor
void print_output1() {
    printf("\nDEPTHWISE OUPUT: \n");
    for (int k = 0; k < DEPTHWISE_FILTER_DEPTH; k++) {
        printf("DEPTH %d:\n", k);
        for (int i = 0; i < OUTPUT1_SIZE; i++) {
            for (int j = 0; j < OUTPUT1_SIZE; j++) {
                printf("%d ", output1[i][j][k]);
            }
            printf("\n");
        }
    }
}

// Print output2 tensor
void print_output2() {
    printf("\nPOINTWISE OUTPUT: \n");
    for (int l = 0; l < POINTWISE_NUM_FILTERS; l++) {
        printf("Filter %d:\n", l);
        for (int k = 0; k < POINTWISE_FILTER_DEPTH; k++) {
            printf("Depth %d:\n", k);
            for (int i = 0; i < OUTPUT2_SIZE; i++) {
                for (int j = 0; j < OUTPUT2_SIZE; j++) {
                    printf("%d ", output2[i][j][k][l]);
                }
                printf("\n");
            }
        }
    }
}

int main() {
    
    // Initialize input 
    init_input();
    // Init conv2d kernel tensor with random values
    init_conv2d_kernel();
    // Initialize depthwise kernel tensor with random values
    init_depthwise_kernel();
    // Initialize pointwise kernel tensor with random values
    init_pointwise_kernel();
    // Print conv2d kernel tensor;
    print_conv2d_kernel();
    // Print depthwise kernel tensor
    print_depthwise_kernel();
    // Print pointwise kernel tensor
    print_pointwise_kernel();
    // Perform conv2d convolution
    conv2d();
    // Print output tensor
    print_output();
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
#include <stdio.h>
#include <stdlib.h>

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr)[0]))

//Input 224x224x3
#define input_size 224
#define input_channels 3
int input[input_size][input_size][input_channels];

//Conv2d_1 112x112x16
#define conv2d_1_kernel_size 3
#define conv2d_1_kernel_channels 3
#define conv2d_1_kernel_num 16
#define conv2d_1_kernel_stride 2
#define conv2d_1_kernel_padding (1.0/2.0)
#define conv2d_1_output_size ((input_size - conv2d_1_kernel_size + 2 * conv2d_1_kernel_padding) / conv2d_1_kernel_stride + 1)
int conv2d_1_output[112][112][16];

//DepthwiseConv2d_1 112x112x16
#define depthwiseconv2d_1_kernel_size 3
#define depthwiseconv2d_1_kernel_channels 16
#define depthwiseconv2d_1_kernel_num 1
#define depthwiseconv2d_1_kernel_stride 1
#define depthwiseconv2d_1_kernel_padding 1
#define depthwiseconv2d_1_output_size ((conv2d_1_output_size - depthwiseconv2d_1_kernel_size + 2 * depthwiseconv2d_1_kernel_padding) / depthwiseconv2d_1_kernel_stride + 1)
int depthwiseconv2d_1_output[112][112][16];

//PointwiseConv2d_1 112x112x16
#define pointwiseconv2d_1_kernel_size 1
#define pointwiseconv2d_1_kernel_channels 16
#define pointwiseconv2d_1_kernel_num 16
#define pointwiseconv2d_1_kernel_stride 1
#define pointwiseconv2d_1_kernel_padding 0
#define pointwiseconv2d_1_output_size ((depthwiseconv2d_1_output_size - pointwiseconv2d_1_kernel_size + 2 * pointwiseconv2d_1_kernel_padding) / pointwiseconv2d_1_kernel_stride + 1)
int pointwiseconv2d_1_output[112][112][16];

//Add_1 112x112x16
#define add_1_output_size 112
#define add_1_output_channels 16
int add_1_output[112][112][16];

//Conv2d_2 112x112x64
#define conv2d_2_kernel_size 1
#define conv2d_2_kernel_channels 16
#define conv2d_2_kernel_num 64
#define conv2d_2_kernel_stride 1
#define conv2d_2_kernel_padding 0
#define conv2d_2_output_size ((add_1_output_size - conv2d_2_kernel_size + 2 * conv2d_2_kernel_padding) / conv2d_2_kernel_stride + 1)
int conv2d_2_output[112][112][64];

//DepthwiseConv2d_2 56x56x64
#define depthwiseconv2d_2_kernel_size 3
#define depthwiseconv2d_2_kernel_channels 64
#define depthwiseconv2d_2_kernel_num 1
#define depthwiseconv2d_2_kernel_stride 2
#define depthwiseconv2d_2_kernel_padding (1.0/2.0)
#define depthwiseconv2d_2_output_size ((conv2d_2_output_size - depthwiseconv2d_2_kernel_size + 2 * depthwiseconv2d_2_kernel_padding) / depthwiseconv2d_2_kernel_stride + 1)
int depthwiseconv2d_2_output[56][56][64];

void init_input() {
    for (int k = 0; k < input_channels; k++){
        // printf("Channel %d\n", k);
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < input_size; j++) {
                input[i][j][k] = rand() % 256;
                // printf("%f ", input[i][j][k]);
            }
            // printf("\n");
        }
    }
}

void conv2d_1() {
    int conv2d_1_kernel[conv2d_1_kernel_size][conv2d_1_kernel_size][conv2d_1_kernel_channels][conv2d_1_kernel_num];
    int conv2d_1_bias[conv2d_1_kernel_num];
    for (int k = 0; k < conv2d_1_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < conv2d_1_kernel_size; i++) {
            for (int j = 0; j < conv2d_1_kernel_size; j++) {
                for (int l = 0; l < conv2d_1_kernel_channels; l++) {
                    conv2d_1_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_1_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }
    for (int k = 0; k < conv2d_1_kernel_num; k++){
        conv2d_1_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_1_bias[k]);
    }
    // Copy input to padded input
    int padded_size = input_size + 2 * conv2d_1_kernel_padding;
    int input_padded[padded_size][padded_size][input_channels];

    // Initilalize padded input with 0
    for (int k = 0; k < input_channels; k++){
        for (int i = 0; i < padded_size; i++) {
            for (int j = 0; j < padded_size; j++) {
                input_padded[i][j][k] = 0;
            }
        }
    }

    // Copy input to padded input
    for (int k = 0; k < input_channels; k++){
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < input_size; j++) {
                input_padded[i][j][k] = input[i][j][k];
            }
        }
    }

    printf("Size of input: %d x %d x %d \n", LEN(input), LEN(input[0]), LEN(input[0][0]));
    printf("Size of padded input: %d x %d x %d \n", LEN(input_padded), LEN(input_padded[0]), LEN(input_padded[0][0]));

    // // Print input
    // for (int k = 0; k < 1; k++){
    //     printf("Channel %d\n", k);
    //     for (int j = 0; j < input_size; j++) {
    //         for (int i = 0; i < input_size; i++) {
    //             printf("%d ", input[i][j][k]);
    //         }
    //         printf("\n");
    //     }
    // }
    // // print padded input
    // for (int k = 0; k < 1; k++){
    //     printf("Channel %d\n", k);
    //     for (int j = 0; j < padded_size; j++) {
    //         for (int i = 0; i < padded_size; i++) {
    //             printf("%d ", input_padded[i][j][k]);
    //         }
    //         printf("\n");
    //     }
    // }

    // Compute Conv2d_1
    for (int k = 0; k < conv2d_1_kernel_num; k++){
        for (int i = 0; i < conv2d_1_output_size; i++) {
            for (int j = 0; j < conv2d_1_output_size; j++) {
                conv2d_1_output[i][j][k] = 0;
                for (int l = 0; l < conv2d_1_kernel_channels; l++) {
                    for (int m = 0; m < conv2d_1_kernel_size; m++) {
                        for (int n = 0; n < conv2d_1_kernel_size; n++) {
                            conv2d_1_output[i][j][k] += input_padded[i*conv2d_1_kernel_stride + m][j*conv2d_1_kernel_stride + n][l] * conv2d_1_kernel[m][n][l][k];
                        }
                    }
                }
                conv2d_1_output[i][j][k] += conv2d_1_bias[k];
            }
        }
    }
    printf("Size of conv2d_1_output: %d x %d x %d \n", LEN(conv2d_1_output), LEN(conv2d_1_output[0]), LEN(conv2d_1_output[0][0]));
    // // print output
    // for (int k = 0; k < 1; k++){
    //     printf("Channel %d\n", k);
    //     for (int j = 0; j < conv2d_1_output_size; j++) {
    //         for (int i = 0; i < conv2d_1_output_size; i++) {
    //             printf("%d ", conv2d_1_output[i][j][k]);
    //         }
    //         printf("\n");
    //     }
    // }
}

void depthwiseconv2d_1(){
    int depthwiseconv2d_1_kernel[depthwiseconv2d_1_kernel_size][depthwiseconv2d_1_kernel_size][depthwiseconv2d_1_kernel_channels][depthwiseconv2d_1_kernel_num];
    int depthwiseconv2d_1_bias[16];
    for (int k = 0; k < depthwiseconv2d_1_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < depthwiseconv2d_1_kernel_size; i++) {
            for (int j = 0; j < depthwiseconv2d_1_kernel_size; j++) {
                for (int l = 0; l < depthwiseconv2d_1_kernel_channels; l++) {
                    depthwiseconv2d_1_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", depthwiseconv2d_1_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }
    for (int k = 0; k < depthwiseconv2d_1_kernel_channels; k++){
        depthwiseconv2d_1_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_1_bias[k]);  
    }

    // Initialize padded conv2d_1_output as 0 array
    int conv2d_1_output_padded_size = conv2d_1_output_size + 2 * depthwiseconv2d_1_kernel_padding;
    int conv2d_1_output_padded[conv2d_1_output_padded_size][conv2d_1_output_padded_size][conv2d_1_kernel_num];
    for (int k = 0; k < depthwiseconv2d_1_kernel_channels; k++){
        for (int i = 0; i < conv2d_1_output_padded_size; i++) {
            for (int j = 0; j < conv2d_1_output_padded_size; j++) {
                conv2d_1_output_padded[i][j][k] = 0;
            }
        }
    }
    // Copy conv2d_1_output to padded conv2d_1_output
    for (int k = 0; k < depthwiseconv2d_1_kernel_channels; k++){
        for (int i = 1; i < conv2d_1_output_size-1; i++) {
            for (int j = 1; j < conv2d_1_output_size-1; j++) {
                conv2d_1_output_padded[i][j][k] = conv2d_1_output[i][j][k];
            }
        }
    }

    // printf("Size of conv2d_1_output: %d x %d x %d \n", LEN(conv2d_1_output), LEN(conv2d_1_output[0]), LEN(conv2d_1_output[0][0]));
    printf("Size of padded conv2d_1_output: %d x %d x %d \n", LEN(conv2d_1_output_padded), LEN(conv2d_1_output_padded[0]), LEN(conv2d_1_output_padded[0][0]));

    // Perform depthwiseconv2d_1
    for (int k = 0; k < depthwiseconv2d_1_kernel_channels; k++){
        for (int i = 0; i < depthwiseconv2d_1_output_size; i++) {
            for (int j = 0; j < depthwiseconv2d_1_output_size; j++) {
                depthwiseconv2d_1_output[i][j][k] = 0;
                for (int l = 0; l < depthwiseconv2d_1_kernel_channels; l++) {
                    for (int m = 0; m < depthwiseconv2d_1_kernel_size; m++) {
                        for (int n = 0; n < depthwiseconv2d_1_kernel_size; n++) {
                            depthwiseconv2d_1_output[i][j][k] += conv2d_1_output_padded[i*depthwiseconv2d_1_kernel_stride + m][j*depthwiseconv2d_1_kernel_stride + n][l] * depthwiseconv2d_1_kernel[m][n][l][k];
                        }
                    }
                }
                depthwiseconv2d_1_output[i][j][k] += depthwiseconv2d_1_bias[k];
            }
        }
    }
    printf("Size of depthwiseconv2d_1_output: %d x %d x %d \n", LEN(depthwiseconv2d_1_output), LEN(depthwiseconv2d_1_output[0]), LEN(depthwiseconv2d_1_output[0][0]));
}

void pointwiseconv2d_1(){
    int pointwiseconv2d_1_kernel[pointwiseconv2d_1_kernel_size][pointwiseconv2d_1_kernel_size][pointwiseconv2d_1_kernel_channels][pointwiseconv2d_1_kernel_num];
    int pointwiseconv2d_1_bias[pointwiseconv2d_1_kernel_num];
    for(int k = 0; k < pointwiseconv2d_1_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < pointwiseconv2d_1_kernel_size; i++) {
            for (int j = 0; j < pointwiseconv2d_1_kernel_size; j++) {
                for (int l = 0; l < pointwiseconv2d_1_kernel_channels; l++) {
                    pointwiseconv2d_1_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_1_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }
    for (int k = 0; k < pointwiseconv2d_1_kernel_num; k++){
        pointwiseconv2d_1_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_1_bias[k]);
    }
    for(int k = 0; k < pointwiseconv2d_1_kernel_num; k++){
        for (int i = 0; i < pointwiseconv2d_1_output_size; i++) {
            for (int j = 0; j < pointwiseconv2d_1_output_size; j++) {
                pointwiseconv2d_1_output[i][j][k] = 0;
                for (int l = 0; l < pointwiseconv2d_1_kernel_channels; l++) {
                    for (int m = 0; m < pointwiseconv2d_1_kernel_size; m++) {
                        for (int n = 0; n < pointwiseconv2d_1_kernel_size; n++) {
                            pointwiseconv2d_1_output[i][j][k] += depthwiseconv2d_1_output[i*pointwiseconv2d_1_kernel_stride + m][j*pointwiseconv2d_1_kernel_stride + n][l] * pointwiseconv2d_1_kernel[m][n][l][k];
                        }
                    }
                }
                pointwiseconv2d_1_output[i][j][k] += pointwiseconv2d_1_bias[k];
            }
        }
    }
    printf("Size of pointwiseconv2d_1_output: %d x %d x %d \n", LEN(pointwiseconv2d_1_output), LEN(pointwiseconv2d_1_output[0]), LEN(pointwiseconv2d_1_output[0][0]));

}

void add_1() {
    for (int i = 0; i < add_1_output_size; i++) {
        for (int j = 0; j < add_1_output_size; j++) {
            for (int k = 0; k < add_1_output_channels; k++) {
                add_1_output[i][j][k] = pointwiseconv2d_1_output[i][j][k] + conv2d_1_output[i][j][k];
                // printf("%f ", add_1_output[i][j][k]);
            }
            // printf("\n");
        }
        // printf("\n");
    }
    printf("Size of add_1_output: %d x %d x %d \n", LEN(add_1_output), LEN(add_1_output[0]), LEN(add_1_output[0][0]));
}

void conv2d_2(){
    int conv2d_2_kernel[conv2d_2_kernel_size][conv2d_2_kernel_size][conv2d_2_kernel_channels][conv2d_2_kernel_num];
    int conv2d_2_bias[conv2d_2_kernel_num];
    for(int k = 0; k < conv2d_2_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < conv2d_2_kernel_size; i++) {
            for (int j = 0; j < conv2d_2_kernel_size; j++) {
                for (int l = 0; l < conv2d_2_kernel_channels; l++) {
                    conv2d_2_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_2_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }
    for (int k = 0; k < conv2d_2_kernel_num; k++){
        conv2d_2_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_2_bias[k]);
    }
    for(int k = 0; k < conv2d_2_kernel_num; k++){
        for (int i = 0; i < conv2d_2_output_size; i++) {
            for (int j = 0; j < conv2d_2_output_size; j++) {
                conv2d_2_output[i][j][k] = 0;
                for (int l = 0; l < conv2d_2_kernel_channels; l++) {
                    for (int m = 0; m < conv2d_2_kernel_size; m++) {
                        for (int n = 0; n < conv2d_2_kernel_size; n++) {
                            conv2d_2_output[i][j][k] += add_1_output[i*conv2d_2_kernel_stride + m][j*conv2d_2_kernel_stride + n][l] * conv2d_2_kernel[m][n][l][k];
                        }
                    }
                }
                conv2d_2_output[i][j][k] += conv2d_2_bias[k];
            }
        }
    }
    printf("Size of conv2d_2_output: %d x %d x %d \n", LEN(conv2d_2_output), LEN(conv2d_2_output[0]), LEN(conv2d_2_output[0][0]));
}

void depthwiseconv2d_2(){
    int depthwiseconv2d_2_kernel[depthwiseconv2d_2_kernel_size][depthwiseconv2d_2_kernel_size][depthwiseconv2d_2_kernel_channels][depthwiseconv2d_2_kernel_num];
    int depthwiseconv2d_2_bias[depthwiseconv2d_2_kernel_num];
    for(int k = 0; k < depthwiseconv2d_2_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < depthwiseconv2d_2_kernel_size; i++) {
            for (int j = 0; j < depthwiseconv2d_2_kernel_size; j++) {
                for (int l = 0; l < depthwiseconv2d_2_kernel_channels; l++) {
                    depthwiseconv2d_2_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", depthwiseconv2d_2_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }
    for (int k = 0; k < depthwiseconv2d_2_kernel_num; k++){
        depthwiseconv2d_2_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_2_bias[k]);
    }

    // Initialize padded conv2d_2_output as 0 array
    int conv2d_2_output_padded_size = conv2d_2_output_size + 2*depthwiseconv2d_2_kernel_padding;
    int conv2d_2_output_padded[conv2d_2_output_padded_size][conv2d_2_output_padded_size][conv2d_2_kernel_channels];
    for(int k = 0; k < conv2d_2_kernel_num; k++){
        for (int i = 0; i < conv2d_2_output_size + 2*depthwiseconv2d_2_kernel_size - 2; i++) {
            for (int j = 0; j < conv2d_2_output_size + 2*depthwiseconv2d_2_kernel_size - 2; j++) {
                conv2d_2_output_padded[i][j][k] = 0;
            }
        }
    }
    for(int k = 0; k < depthwiseconv2d_2_kernel_num; k++){
        for (int i = 0; i < depthwiseconv2d_2_output_size; i++) {
            for (int j = 0; j < depthwiseconv2d_2_output_size; j++) {
                depthwiseconv2d_2_output[i][j][k] = 0;
                for (int l = 0; l < depthwiseconv2d_2_kernel_channels; l++) {
                    for (int m = 0; m < depthwiseconv2d_2_kernel_size; m++) {
                        for (int n = 0; n < depthwiseconv2d_2_kernel_size; n++) {
                            depthwiseconv2d_2_output[i][j][k] += conv2d_2_output[i*depthwiseconv2d_2_kernel_stride + m][j*depthwiseconv2d_2_kernel_stride + n][l] * depthwiseconv2d_2_kernel[m][n][l][k];
                        }
                    }
                }
                depthwiseconv2d_2_output[i][j][k] += depthwiseconv2d_2_bias[k];
            }
        }
    }
}



int main (){
    init_input();
    conv2d_1();
    depthwiseconv2d_1();
    pointwiseconv2d_1();
    add_1();
    conv2d_2();

    return 0;
}
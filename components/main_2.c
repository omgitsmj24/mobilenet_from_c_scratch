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

//PointwiseConv2d_2 56x56x24
#define pointwiseconv2d_2_kernel_size 1
#define pointwiseconv2d_2_kernel_channels 64
#define pointwiseconv2d_2_kernel_num 24
#define pointwiseconv2d_2_kernel_stride 1
#define pointwiseconv2d_2_kernel_padding 0
#define pointwiseconv2d_2_output_size ((depthwiseconv2d_2_output_size - pointwiseconv2d_2_kernel_size + 2 * pointwiseconv2d_2_kernel_padding) / pointwiseconv2d_2_kernel_stride + 1)
int pointwiseconv2d_2_output[56][56][24];

//Conv2d_3 56x56x72
#define conv2d_3_kernel_size 1
#define conv2d_3_kernel_channels 24
#define conv2d_3_kernel_num 72
#define conv2d_3_kernel_stride 1
#define conv2d_3_kernel_padding 0
#define conv2d_3_output_size ((pointwiseconv2d_2_output_size - conv2d_3_kernel_size + 2 * conv2d_3_kernel_padding) / conv2d_3_kernel_stride + 1)
int conv2d_3_output[56][56][72];

//DepthwiseConv2d_3 56x56x72
#define depthwiseconv2d_3_kernel_size 3
#define depthwiseconv2d_3_kernel_channels 72
#define depthwiseconv2d_3_kernel_num 1
#define depthwiseconv2d_3_kernel_stride 1
#define depthwiseconv2d_3_kernel_padding 1
#define depthwiseconv2d_3_output_size ((conv2d_3_output_size - depthwiseconv2d_3_kernel_size + 2 * depthwiseconv2d_3_kernel_padding) / depthwiseconv2d_3_kernel_stride + 1)
int depthwiseconv2d_3_output[56][56][72];

//PointwiseConv2d_3 56x56x24
#define pointwiseconv2d_3_kernel_size 1
#define pointwiseconv2d_3_kernel_channels 72
#define pointwiseconv2d_3_kernel_num 24
#define pointwiseconv2d_3_kernel_stride 1
#define pointwiseconv2d_3_kernel_padding 0
#define pointwiseconv2d_3_output_size ((depthwiseconv2d_3_output_size - pointwiseconv2d_3_kernel_size + 2 * pointwiseconv2d_3_kernel_padding) / pointwiseconv2d_3_kernel_stride + 1)
int pointwiseconv2d_3_output[56][56][24];

//Add_2 56x56x24
#define add_2_output_size 56
#define add_2_output_channels 24
int add_2_output[56][56][24];

//Conv2d_4 56x56x72
#define conv2d_4_kernel_size 1
#define conv2d_4_kernel_channels 24
#define conv2d_4_kernel_num 72
#define conv2d_4_kernel_stride 1
#define conv2d_4_kernel_padding 0
#define conv2d_4_output_size ((add_2_output_size - conv2d_4_kernel_size + 2 * conv2d_4_kernel_padding) / conv2d_4_kernel_stride + 1)
int conv2d_4_output[56][56][72];

//DepthwiseConv2d_4 28x28x72
#define depthwiseconv2d_4_kernel_size 3
#define depthwiseconv2d_4_kernel_channels 72
#define depthwiseconv2d_4_kernel_num 1
#define depthwiseconv2d_4_kernel_stride 2
#define depthwiseconv2d_4_kernel_padding (1.0/2.0)
#define depthwiseconv2d_4_output_size ((conv2d_4_output_size - depthwiseconv2d_4_kernel_size + 2 * depthwiseconv2d_4_kernel_padding) / depthwiseconv2d_4_kernel_stride + 1)
int depthwiseconv2d_4_output[28][28][72];

//PointwiseConv2d_4 28x28x40
#define pointwiseconv2d_4_kernel_size 1
#define pointwiseconv2d_4_kernel_channels 72
#define pointwiseconv2d_4_kernel_num 40
#define pointwiseconv2d_4_kernel_stride 1
#define pointwiseconv2d_4_kernel_padding 0
#define pointwiseconv2d_4_output_size ((depthwiseconv2d_4_output_size - pointwiseconv2d_4_kernel_size + 2 * pointwiseconv2d_4_kernel_padding) / pointwiseconv2d_4_kernel_stride + 1)
int pointwiseconv2d_4_output[28][28][40];

//Conv2d_5 28x28x120
#define conv2d_5_kernel_size 1
#define conv2d_5_kernel_channels 40
#define conv2d_5_kernel_num 120
#define conv2d_5_kernel_stride 1
#define conv2d_5_kernel_padding 0
#define conv2d_5_output_size ((pointwiseconv2d_4_output_size - conv2d_5_kernel_size + 2 * conv2d_5_kernel_padding) / conv2d_5_kernel_stride + 1)
int conv2d_5_output[28][28][120];

//DepthwiseConv2d_5 28x28x120
#define depthwiseconv2d_5_kernel_size 3
#define depthwiseconv2d_5_kernel_channels 120
#define depthwiseconv2d_5_kernel_num 1
#define depthwiseconv2d_5_kernel_stride 1
#define depthwiseconv2d_5_kernel_padding 1
#define depthwiseconv2d_5_output_size ((conv2d_5_output_size - depthwiseconv2d_5_kernel_size + 2 * depthwiseconv2d_5_kernel_padding) / depthwiseconv2d_5_kernel_stride + 1)
int depthwiseconv2d_5_output[28][28][120];

//PointwiseConv2d_5 28x28x40
#define pointwiseconv2d_5_kernel_size 1
#define pointwiseconv2d_5_kernel_channels 120
#define pointwiseconv2d_5_kernel_num 40
#define pointwiseconv2d_5_kernel_stride 1
#define pointwiseconv2d_5_kernel_padding 0
#define pointwiseconv2d_5_output_size ((depthwiseconv2d_5_output_size - pointwiseconv2d_5_kernel_size + 2 * pointwiseconv2d_5_kernel_padding) / pointwiseconv2d_5_kernel_stride + 1)
int pointwiseconv2d_5_output[28][28][40];

//Add_3 28x28x40 
#define add_3_output_size 28
#define add_3_output_channels 40
int add_3_output[28][28][40];

//Conv2d_6 28x28x120
#define conv2d_6_kernel_size 1
#define conv2d_6_kernel_channels 40
#define conv2d_6_kernel_num 120
#define conv2d_6_kernel_stride 1
#define conv2d_6_kernel_padding 0
#define conv2d_6_output_size ((add_3_output_size - conv2d_6_kernel_size + 2 * conv2d_6_kernel_padding) / conv2d_6_kernel_stride + 1)
int conv2d_6_output[28][28][120];

//DepthwiseConv2d_6 28x28x120
#define depthwiseconv2d_6_kernel_size 3
#define depthwiseconv2d_6_kernel_channels 120
#define depthwiseconv2d_6_kernel_num 1
#define depthwiseconv2d_6_kernel_stride 1
#define depthwiseconv2d_6_kernel_padding 1
#define depthwiseconv2d_6_output_size ((conv2d_6_output_size - depthwiseconv2d_6_kernel_size + 2 * depthwiseconv2d_6_kernel_padding) / depthwiseconv2d_6_kernel_stride + 1)
int depthwiseconv2d_6_output[28][28][120];

//PointwiseConv2d_6 28x28x40
#define pointwiseconv2d_6_kernel_size 1
#define pointwiseconv2d_6_kernel_channels 120
#define pointwiseconv2d_6_kernel_num 40
#define pointwiseconv2d_6_kernel_stride 1
#define pointwiseconv2d_6_kernel_padding 0
#define pointwiseconv2d_6_output_size ((depthwiseconv2d_6_output_size - pointwiseconv2d_6_kernel_size + 2 * pointwiseconv2d_6_kernel_padding) / pointwiseconv2d_6_kernel_stride + 1)
int pointwiseconv2d_6_output[28][28][40];

//Add_4 28x28x40
#define add_4_output_size 28
#define add_4_output_channels 40
int add_4_output[28][28][40];

//Conv2d_7 28x28x240
#define conv2d_7_kernel_size 1
#define conv2d_7_kernel_channels 40
#define conv2d_7_kernel_num 240
#define conv2d_7_kernel_stride 1
#define conv2d_7_kernel_padding 0
#define conv2d_7_output_size ((add_4_output_size - conv2d_7_kernel_size + 2 * conv2d_7_kernel_padding) / conv2d_7_kernel_stride + 1)
int conv2d_7_output[28][28][240];

//DepthwiseConv2d_7 14x14x240
#define depthwiseconv2d_7_kernel_size 3
#define depthwiseconv2d_7_kernel_channels 240
#define depthwiseconv2d_7_kernel_num 1
#define depthwiseconv2d_7_kernel_stride 2
#define depthwiseconv2d_7_kernel_padding (1.0/2.0)
#define depthwiseconv2d_7_output_size ((conv2d_7_output_size - depthwiseconv2d_7_kernel_size + 2 * depthwiseconv2d_7_kernel_padding) / depthwiseconv2d_7_kernel_stride + 1)
int depthwiseconv2d_7_output[14][14][240];

//PointwiseConv2d_7 14x14x80
#define pointwiseconv2d_7_kernel_size 1
#define pointwiseconv2d_7_kernel_channels 240
#define pointwiseconv2d_7_kernel_num 80
#define pointwiseconv2d_7_kernel_stride 1
#define pointwiseconv2d_7_kernel_padding 0
#define pointwiseconv2d_7_output_size ((depthwiseconv2d_7_output_size - pointwiseconv2d_7_kernel_size + 2 * pointwiseconv2d_7_kernel_padding) / pointwiseconv2d_7_kernel_stride + 1)
int pointwiseconv2d_7_output[14][14][80];

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
    int depthwiseconv2d_2_bias[depthwiseconv2d_2_kernel_channels];
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

    // Initialize bias
    for (int k = 0; k < depthwiseconv2d_2_kernel_channels; k++){
        depthwiseconv2d_2_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_2_bias[k]);
    }

    // Initialize padded conv2d_2_output as 0 array
    int conv2d_2_output_padded_size = conv2d_2_output_size + 2*depthwiseconv2d_2_kernel_padding;
    int conv2d_2_output_padded[conv2d_2_output_padded_size][conv2d_2_output_padded_size][conv2d_2_kernel_num];
    for(int k = 0; k < conv2d_2_kernel_num; k++){
        for (int i = 0; i < conv2d_2_output_padded_size; i++) {
            for (int j = 0; j < conv2d_2_output_padded_size; j++) {
                conv2d_2_output_padded[i][j][k] = 0;
            }
        }
    }

    // Copy conv2d_2_output to padded conv2d_2_output
    for(int k = 0; k < conv2d_2_kernel_num; k++){
        for (int i = 0; i < conv2d_2_output_size; i++) {
            for (int j = 0; j < conv2d_2_output_size; j++) {
                conv2d_2_output_padded[i][j][k] = conv2d_2_output[i][j][k];
            }
        }
    }

    printf("Size of padded conv2d_2_output: %d x %d x %d \n", LEN(conv2d_2_output_padded), LEN(conv2d_2_output_padded[0]), LEN(conv2d_2_output_padded[0][0]));

    // Perform depthwiseconv2d_2
    for(int k = 0; k < depthwiseconv2d_2_kernel_num; k++){
        for (int i = 0; i < depthwiseconv2d_2_output_size; i++) {
            for (int j = 0; j < depthwiseconv2d_2_output_size; j++) {
                for (int l = 0; l < depthwiseconv2d_2_kernel_channels; l++) {
                    depthwiseconv2d_2_output[i][j][l] = 0;
                    for (int m = 0; m < depthwiseconv2d_2_kernel_size; m++) {
                        for (int n = 0; n < depthwiseconv2d_2_kernel_size; n++) {
                            depthwiseconv2d_2_output[i][j][l] += conv2d_2_output_padded[i*depthwiseconv2d_2_kernel_stride + m][j*depthwiseconv2d_2_kernel_stride + n][l] * depthwiseconv2d_2_kernel[m][n][l][k];
                        }
                    }
                    depthwiseconv2d_2_output[i][j][l] += depthwiseconv2d_2_bias[l];
                }
            }
        }
    }


    printf("Size of depthwiseconv2d_2_output: %d x %d x %d \n", LEN(depthwiseconv2d_2_output), LEN(depthwiseconv2d_2_output[0]), LEN(depthwiseconv2d_2_output[0][0]));
}

void pointwiseconv2d_2(){
    int pointwiseconv2d_2_kernel[pointwiseconv2d_2_kernel_size][pointwiseconv2d_2_kernel_size][pointwiseconv2d_2_kernel_channels][pointwiseconv2d_2_kernel_num];
    int pointwiseconv2d_2_bias[pointwiseconv2d_2_kernel_num];
    for(int k = 0; k < pointwiseconv2d_2_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < pointwiseconv2d_2_kernel_size; i++) {
            for (int j = 0; j < pointwiseconv2d_2_kernel_size; j++) {
                for (int l = 0; l < pointwiseconv2d_2_kernel_channels; l++) {
                    pointwiseconv2d_2_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_2_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for (int k = 0; k < pointwiseconv2d_2_kernel_num; k++){
        pointwiseconv2d_2_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_2_bias[k]);
    }

    // Perform pointwiseconv2d_2
    for(int k = 0; k < pointwiseconv2d_2_kernel_num; k++){
        for (int i = 0; i < pointwiseconv2d_2_output_size; i++) {
            for (int j = 0; j < pointwiseconv2d_2_output_size; j++) {
                pointwiseconv2d_2_output[i][j][k] = 0;
                for (int l = 0; l < pointwiseconv2d_2_kernel_channels; l++) {
                    for (int m = 0; m < pointwiseconv2d_2_kernel_size; m++) {
                        for (int n = 0; n < pointwiseconv2d_2_kernel_size; n++) {
                            pointwiseconv2d_2_output[i][j][k] += depthwiseconv2d_2_output[i*pointwiseconv2d_2_kernel_stride + m][j*pointwiseconv2d_2_kernel_stride + n][l] * pointwiseconv2d_2_kernel[m][n][l][k];
                        }
                    }
                }
                pointwiseconv2d_2_output[i][j][k] += pointwiseconv2d_2_bias[k];
            }
        }
    }

    printf("Size of pointwiseconv2d_2_output: %d x %d x %d \n", LEN(pointwiseconv2d_2_output), LEN(pointwiseconv2d_2_output[0]), LEN(pointwiseconv2d_2_output[0][0]));
}

void conv2d_3(){
    int conv2d_3_kernel[conv2d_3_kernel_size][conv2d_3_kernel_size][conv2d_3_kernel_channels][conv2d_3_kernel_num];
    int conv2d_3_bias[conv2d_3_kernel_num];
    for(int k = 0; k < conv2d_3_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < conv2d_3_kernel_size; i++) {
            for (int j = 0; j < conv2d_3_kernel_size; j++) {
                for (int l = 0; l < conv2d_3_kernel_channels; l++) {
                    conv2d_3_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_3_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for (int k = 0; k < conv2d_3_kernel_num; k++){
        conv2d_3_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_3_bias[k]);
    }

    // Perform conv2d_3
    for(int k = 0; k < conv2d_3_kernel_num; k++){
        for (int i = 0; i < conv2d_3_output_size; i++) {
            for (int j = 0; j < conv2d_3_output_size; j++) {
                conv2d_3_output[i][j][k] = 0;
                for (int l = 0; l < conv2d_3_kernel_channels; l++) {
                    for (int m = 0; m < conv2d_3_kernel_size; m++) {
                        for (int n = 0; n < conv2d_3_kernel_size; n++) {
                            conv2d_3_output[i][j][k] += pointwiseconv2d_2_output[i*conv2d_3_kernel_stride + m][j*conv2d_3_kernel_stride + n][l] * conv2d_3_kernel[m][n][l][k];
                        }
                    }
                }
                conv2d_3_output[i][j][k] += conv2d_3_bias[k];
            }
        }
    }
    printf("Size of conv2d_3_output: %d x %d x %d \n", LEN(conv2d_3_output), LEN(conv2d_3_output[0]), LEN(conv2d_3_output[0][0]));
}

void depthwiseconv2d_3(){
    int depthwiseconv2d_3_kernel[depthwiseconv2d_3_kernel_size][depthwiseconv2d_3_kernel_size][depthwiseconv2d_3_kernel_channels];
    int depthwiseconv2d_3_bias[depthwiseconv2d_3_kernel_channels];
    for(int k = 0; k < depthwiseconv2d_3_kernel_channels; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < depthwiseconv2d_3_kernel_size; i++) {
            for (int j = 0; j < depthwiseconv2d_3_kernel_size; j++) {
                depthwiseconv2d_3_kernel[i][j][k] = rand() % 5 - 2;
                // printf("%f ", depthwiseconv2d_3_kernel[i][j][k]);
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for (int k = 0; k < depthwiseconv2d_3_kernel_channels; k++){
        depthwiseconv2d_3_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_3_bias[k]);
    }

    // Initialize padded conv2d_3_output with zero value
    int padded_conv2d_3_output_size = conv2d_3_output_size + depthwiseconv2d_3_kernel_padding*2;
    int padded_conv2d_3_output[padded_conv2d_3_output_size][padded_conv2d_3_output_size][conv2d_3_kernel_num];
    for (int i = 0; i < padded_conv2d_3_output_size; i++) {
        for (int j = 0; j < padded_conv2d_3_output_size; j++) {
            for (int k = 0; k < conv2d_3_kernel_num; k++) {
                padded_conv2d_3_output[i][j][k] = 0;
                }
            }
        }

    // Copy conv2d_3_output to padded_conv2d_3_output
    for (int i = 0; i < conv2d_3_output_size; i++) {
        for (int j = 0; j < conv2d_3_output_size; j++) {
            for (int k = 0; k < conv2d_3_kernel_num; k++) {
                padded_conv2d_3_output[i][j][k] = conv2d_3_output[i][j][k];
            }
        }
    }

    printf("Size of padded_conv2d_3_output: %d x %d x %d \n", LEN(padded_conv2d_3_output), LEN(padded_conv2d_3_output[0]), LEN(padded_conv2d_3_output[0][0]));

    // Perform depthwiseconv2d_3
    for (int k = 0; k < depthwiseconv2d_3_kernel_channels; k++) {
        for (int i = 0; i < depthwiseconv2d_3_output_size; i++) {
            for (int j = 0; j < depthwiseconv2d_3_output_size; j++) {
                depthwiseconv2d_3_output[i][j][k] = 0;
                for (int l = 0; l < depthwiseconv2d_3_kernel_size; l++) {
                    for (int m = 0; m < depthwiseconv2d_3_kernel_size; m++) {
                        depthwiseconv2d_3_output[i][j][k] += padded_conv2d_3_output[i*depthwiseconv2d_3_kernel_stride + l][j*depthwiseconv2d_3_kernel_stride + m][k] * depthwiseconv2d_3_kernel[l][m][k];
                    }
                }
                depthwiseconv2d_3_output[i][j][k] += depthwiseconv2d_3_bias[k];
            }
        }
    }
    
    printf("Size of depthwiseconv2d_3_output: %d x %d x %d \n", LEN(depthwiseconv2d_3_output), LEN(depthwiseconv2d_3_output[0]), LEN(depthwiseconv2d_3_output[0][0]));
}

void pointwiseconv2d_3(){
    int pointwiseconv2d_3_kernel[pointwiseconv2d_3_kernel_size][pointwiseconv2d_3_kernel_size][pointwiseconv2d_3_kernel_channels][pointwiseconv2d_3_kernel_num];
    int pointwiseconv2d_3_bias[pointwiseconv2d_3_kernel_num];
    for(int k = 0; k < pointwiseconv2d_3_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < pointwiseconv2d_3_kernel_size; i++) {
            for (int j = 0; j < pointwiseconv2d_3_kernel_size; j++) {
                for (int l = 0; l < pointwiseconv2d_3_kernel_channels; l++) {
                    pointwiseconv2d_3_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_3_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for (int k = 0; k < pointwiseconv2d_3_kernel_num; k++){
        pointwiseconv2d_3_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_3_bias[k]);
    }

    //Perform pointwiseconv2d_3
    for(int k = 0; k < pointwiseconv2d_3_kernel_num; k++){
        for (int i = 0; i < pointwiseconv2d_3_output_size; i++) {
            for (int j = 0; j < pointwiseconv2d_3_output_size; j++) {
                pointwiseconv2d_3_output[i][j][k] = 0;
                for (int l = 0; l < pointwiseconv2d_3_kernel_channels; l++) {
                    for (int m = 0; m < pointwiseconv2d_3_kernel_size; m++) {
                        for (int n = 0; n < pointwiseconv2d_3_kernel_size; n++) {
                            pointwiseconv2d_3_output[i][j][k] += depthwiseconv2d_3_output[i*pointwiseconv2d_3_kernel_stride + m][j*pointwiseconv2d_3_kernel_stride + n][l] * pointwiseconv2d_3_kernel[m][n][l][k];
                        }
                    }
                }
                pointwiseconv2d_3_output[i][j][k] += pointwiseconv2d_3_bias[k];
            }
        }
    }
    printf("Size of pointwiseconv2d_3_output: %d x %d x %d \n", LEN(pointwiseconv2d_3_output), LEN(pointwiseconv2d_3_output[0]), LEN(pointwiseconv2d_3_output[0][0]));
}

void add_2(){
    for (int i = 0; i < add_2_output_size; i++) {
        for (int j = 0; j < add_2_output_size; j++) {
            for (int k = 0; k < add_2_output_channels; k++) {
                add_2_output[i][j][k] = pointwiseconv2d_3_output[i][j][k] + pointwiseconv2d_2_output[i][j][k];
            }
        }
    }
    printf("Size of add_2_output: %d x %d x %d \n", LEN(add_2_output), LEN(add_2_output[0]), LEN(add_2_output[0][0]));
}

void conv2d_4(){
    int conv2d_4_kernel[conv2d_4_kernel_size][conv2d_4_kernel_size][conv2d_4_kernel_channels][conv2d_4_kernel_num];
    int conv2d_4_bias[conv2d_4_kernel_num];
    for(int k = 0; k < conv2d_4_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < conv2d_4_kernel_size; i++) {
            for (int j = 0; j < conv2d_4_kernel_size; j++) {
                for (int l = 0; l < conv2d_4_kernel_channels; l++) {
                    conv2d_4_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_4_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for (int k = 0; k < conv2d_4_kernel_num; k++){
        conv2d_4_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_4_bias[k]);
    }

    //Perform conv2d_4
    for(int k = 0; k < conv2d_4_kernel_num; k++){
        for (int i = 0; i < conv2d_4_output_size; i++) {
            for (int j = 0; j < conv2d_4_output_size; j++) {
                conv2d_4_output[i][j][k] = 0;
                for (int l = 0; l < conv2d_4_kernel_channels; l++) {
                    for (int m = 0; m < conv2d_4_kernel_size; m++) {
                        for (int n = 0; n < conv2d_4_kernel_size; n++) {
                            conv2d_4_output[i][j][k] += add_2_output[i*conv2d_4_kernel_stride + m][j*conv2d_4_kernel_stride + n][l] * conv2d_4_kernel[m][n][l][k];
                        }
                    }
                }
                conv2d_4_output[i][j][k] += conv2d_4_bias[k];
            }
        }
    }
    printf("Size of conv2d_4_output: %d x %d x %d \n", LEN(conv2d_4_output), LEN(conv2d_4_output[0]), LEN(conv2d_4_output[0][0]));
}

void depthwiseconv2d_4(){
    int depthwiseconv2d_4_kernel[depthwiseconv2d_4_kernel_size][depthwiseconv2d_4_kernel_size][depthwiseconv2d_4_kernel_channels];
    int depthwiseconv2d_4_bias[depthwiseconv2d_4_kernel_channels];
    for (int k = 0; k < depthwiseconv2d_4_kernel_channels; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < depthwiseconv2d_4_kernel_size; i++) {
            for (int j = 0; j < depthwiseconv2d_4_kernel_size; j++) {
                depthwiseconv2d_4_kernel[i][j][k] = rand() % 5 - 2;
                // printf("%f ", depthwiseconv2d_4_kernel[i][j][k]);
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for (int k = 0; k < depthwiseconv2d_4_kernel_channels; k++){
        depthwiseconv2d_4_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_4_bias[k]);
    }

    // Determine padded conv2d_4_output size
    int conv2d_4_output_padded_size = conv2d_4_output_size + depthwiseconv2d_4_kernel_padding * 2;
    int conv2d_4_output_padded[conv2d_4_output_padded_size][conv2d_4_output_padded_size][conv2d_4_kernel_num];
    // Initialize padded conv2d_4_output with zero
    for (int k = 0; k < depthwiseconv2d_4_kernel_channels; k++) {
        for (int i = 0; i < conv2d_4_output_padded_size; i++) {
            for (int j = 0; j < conv2d_4_output_padded_size; j++) {
                conv2d_4_output[i][j][k] = 0;
            }
        }
    }

    // Copy conv2d_4_output to padded conv2d_4_output
    for (int k = 0; k < depthwiseconv2d_4_kernel_channels; k++) {
        for (int i = 0; i < conv2d_4_output_size; i++) {
            for (int j = 0; j < conv2d_4_output_size; j++) {
                conv2d_4_output_padded[i][j][k] = conv2d_4_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_4_output_padded: %d x %d x %d \n", LEN(conv2d_4_output_padded), LEN(conv2d_4_output_padded[0]), LEN(conv2d_4_output_padded[0][0]));

    //Perform depthwiseconv2d_4
    for (int k = 0; k < depthwiseconv2d_4_kernel_channels; k++) {
        for (int i = 0; i < depthwiseconv2d_4_output_size; i++) {
            for (int j = 0; j < depthwiseconv2d_4_output_size; j++) {
                depthwiseconv2d_4_output[i][j][k] = 0;
                for (int l = 0; l < depthwiseconv2d_4_kernel_size; l++) {
                    for (int m = 0; m < depthwiseconv2d_4_kernel_size; m++) {
                        depthwiseconv2d_4_output[i][j][k] += conv2d_4_output_padded[i*depthwiseconv2d_4_kernel_stride + l][j*depthwiseconv2d_4_kernel_stride + m][k] * depthwiseconv2d_4_kernel[l][m][k];
                    }
                }
                depthwiseconv2d_4_output[i][j][k] += depthwiseconv2d_4_bias[k];
            }
        }
    }

    printf("Size of depthwiseconv2d_4_output: %d x %d x %d \n", LEN(depthwiseconv2d_4_output), LEN(depthwiseconv2d_4_output[0]), LEN(depthwiseconv2d_4_output[0][0]));
}

void pointwiseconv2d_4(){
    int pointwiseconv2d_4_kernel[pointwiseconv2d_4_kernel_size][pointwiseconv2d_4_kernel_size][pointwiseconv2d_4_kernel_channels][pointwiseconv2d_4_kernel_num];
    int pointwiseconv2d_4_bias[pointwiseconv2d_4_kernel_num];
    for (int k = 0; k < pointwiseconv2d_4_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int i = 0; i < pointwiseconv2d_4_kernel_size; i++) {
            for (int j = 0; j < pointwiseconv2d_4_kernel_size; j++) {
                for (int l = 0; l < pointwiseconv2d_4_kernel_channels; l++) {
                    pointwiseconv2d_4_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_4_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for (int k = 0; k < pointwiseconv2d_4_kernel_num; k++){
        pointwiseconv2d_4_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_4_bias[k]);
    }

    // Perform pointwiseconv2d_4
    for (int k = 0; k < pointwiseconv2d_4_kernel_num; k++) {
        for (int i = 0; i < pointwiseconv2d_4_output_size; i++) {
            for (int j = 0; j < pointwiseconv2d_4_output_size; j++) {
                pointwiseconv2d_4_output[i][j][k] = 0;
                for (int l = 0; l < pointwiseconv2d_4_kernel_channels; l++) {
                    for (int m = 0; m < pointwiseconv2d_4_kernel_size; m++) {
                        for (int n = 0; n < pointwiseconv2d_4_kernel_size; n++) {
                            pointwiseconv2d_4_output[i][j][k] += depthwiseconv2d_4_output[i][j][l] * pointwiseconv2d_4_kernel[m][n][l][k];
                        }
                    }
                }
                pointwiseconv2d_4_output[i][j][k] += pointwiseconv2d_4_bias[k];
            }
        }
    }

    printf("Size of pointwiseconv2d_4_output: %d x %d x %d \n", LEN(pointwiseconv2d_4_output), LEN(pointwiseconv2d_4_output[0]), LEN(pointwiseconv2d_4_output[0][0]));
}

void conv2d_5(){
    int conv2d_5_kernel[conv2d_5_kernel_size][conv2d_5_kernel_size][conv2d_5_kernel_channels][conv2d_5_kernel_num];
    int conv2d_5_bias[conv2d_5_kernel_num];
    for(int k = 0; k < conv2d_5_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for(int i = 0; i < conv2d_5_kernel_size; i++){
            for(int j = 0; j < conv2d_5_kernel_size; j++){
                for(int l = 0; l < conv2d_5_kernel_channels; l++){
                    conv2d_5_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_5_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for(int k = 0; k < conv2d_5_kernel_num; k++){
        conv2d_5_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_5_bias[k]);
    }

    //Perform conv2d_5
    for(int k = 0; k < conv2d_5_kernel_num; k++){
        for(int i = 0; i < conv2d_5_output_size; i++){
            for(int j = 0; j < conv2d_5_output_size; j++){
                conv2d_5_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_5_kernel_channels; l++){
                    for(int m = 0; m < conv2d_5_kernel_size; m++){
                        for(int n = 0; n < conv2d_5_kernel_size; n++){
                            conv2d_5_output[i][j][k] += pointwiseconv2d_4_output[i*conv2d_5_kernel_stride + m][j*conv2d_5_kernel_stride + n][l] * conv2d_5_kernel[m][n][l][k];
                        }
                    }
                }
                conv2d_5_output[i][j][k] += conv2d_5_bias[k];
            }
        }
    }

    printf("Size of conv2d_5_output: %d x %d x %d \n", LEN(conv2d_5_output), LEN(conv2d_5_output[0]), LEN(conv2d_5_output[0][0]));
}

void depthwiseconv2d_5(){
    int depthwiseconv2d_5_kernel[depthwiseconv2d_5_kernel_size][depthwiseconv2d_5_kernel_size][depthwiseconv2d_5_kernel_channels][depthwiseconv2d_5_kernel_num];
    int depthwiseconv2d_5_bias[depthwiseconv2d_5_kernel_num];
    for(int k = 0; k < depthwiseconv2d_5_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for(int i = 0; i < depthwiseconv2d_5_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_5_kernel_size; j++){
                for(int l = 0; l < depthwiseconv2d_5_kernel_channels; l++){
                    depthwiseconv2d_5_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", depthwiseconv2d_5_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_5_kernel_channels; k++){
        depthwiseconv2d_5_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_5_bias[k]);
    }

    // Determine padded conv2d_5_output size
    int conv2d_5_padded_output_size = depthwiseconv2d_5_output_size + 2*depthwiseconv2d_1_kernel_padding;
    int conv2d_5_padded_output[conv2d_5_padded_output_size][conv2d_5_padded_output_size][conv2d_5_kernel_num];
    // Initialize padded conv2d_5_output with 0
    for(int i = 0; i < conv2d_5_padded_output_size; i++){
        for(int j = 0; j < conv2d_5_padded_output_size; j++){
            for(int k = 0; k < depthwiseconv2d_5_kernel_channels; k++){
                conv2d_5_padded_output[i][j][k] = 0;
            }
        }
    }
    // Copy conv2d_5_output to padded conv2d_5_output
    for(int i = 1; i < depthwiseconv2d_5_output_size - 1; i++){
        for(int j = 1; j < depthwiseconv2d_5_output_size - 1; j++){
            for(int k = 0; k < depthwiseconv2d_5_kernel_channels; k++){
                conv2d_5_padded_output[i][j][k] = conv2d_5_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_5_padded_output: %d x %d x %d \n", LEN(conv2d_5_padded_output), LEN(conv2d_5_padded_output[0]), LEN(conv2d_5_padded_output[0][0]));

    //Perform depthwiseconv2d_5
    for (int k = 0; k < depthwiseconv2d_5_kernel_channels; k++){
        for(int i = 0; i < depthwiseconv2d_5_output_size; i++){
            for(int j = 0; j < depthwiseconv2d_5_output_size; j++){
                depthwiseconv2d_5_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_5_kernel_channels; l++){
                    for(int m = 0; m < depthwiseconv2d_5_kernel_size; m++){
                        for(int n = 0; n < depthwiseconv2d_5_kernel_size; n++){
                            depthwiseconv2d_5_output[i][j][l] += conv2d_5_padded_output[i*depthwiseconv2d_5_kernel_stride + m][j*depthwiseconv2d_5_kernel_stride + n][l] * depthwiseconv2d_5_kernel[m][n][l][k];
                        }
                    }
                    depthwiseconv2d_5_output[i][j][k] += depthwiseconv2d_5_bias[l];
                }
            }
        }
    }

    printf("Size of depthwiseconv2d_5_output: %d x %d x %d \n", LEN(depthwiseconv2d_5_output), LEN(depthwiseconv2d_5_output[0]), LEN(depthwiseconv2d_5_output[0][0]));
}

void pointwiseconv2d_5(){
    int pointwiseconv2d_5_kernel[pointwiseconv2d_5_kernel_size][pointwiseconv2d_5_kernel_size][pointwiseconv2d_5_kernel_channels][pointwiseconv2d_5_kernel_num];
    int pointwiseconv2d_5_bias[pointwiseconv2d_5_kernel_num];
    for(int k = 0; k < pointwiseconv2d_5_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for(int i = 0; i < pointwiseconv2d_5_kernel_size; i++){
            for(int j = 0; j < pointwiseconv2d_5_kernel_size; j++){
                for(int l = 0; l < pointwiseconv2d_5_kernel_channels; l++){
                    pointwiseconv2d_5_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_5_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_5_kernel_num; k++){
        pointwiseconv2d_5_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_5_bias[k]);
    }

    // Perform pointwiseconv2d_5
    for(int k = 0; k < pointwiseconv2d_5_kernel_num; k++){
        for(int i = 0; i < depthwiseconv2d_5_output_size; i++){
            for(int j = 0; j < depthwiseconv2d_5_output_size; j++){
                pointwiseconv2d_5_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_5_kernel_channels; l++){
                    for(int m = 0; m < pointwiseconv2d_5_kernel_size; m++){
                        for(int n = 0; n < pointwiseconv2d_5_kernel_size; n++){
                            pointwiseconv2d_5_output[i][j][k] += depthwiseconv2d_5_output[i*pointwiseconv2d_5_kernel_stride + m][j*pointwiseconv2d_5_kernel_stride + n][l] * pointwiseconv2d_5_kernel[m][n][l][k];
                        }
                    }
                    pointwiseconv2d_5_output[i][j][k] += pointwiseconv2d_5_bias[k];
                }
            }
        }
    }

    printf("Size of pointwiseconv2d_5_output: %d x %d x %d \n", LEN(pointwiseconv2d_5_output), LEN(pointwiseconv2d_5_output[0]), LEN(pointwiseconv2d_5_output[0][0]));
}

void add_3(){
    for(int i = 0; i < depthwiseconv2d_5_output_size; i++){
        for(int j = 0; j < depthwiseconv2d_5_output_size; j++){
            for(int k = 0; k < depthwiseconv2d_5_kernel_channels; k++){
                add_3_output[i][j][k] = pointwiseconv2d_5_output[i][j][k] + pointwiseconv2d_4_output[i][j][k];
            }
        }
    }

    printf("Size of add_3_output: %d x %d x %d \n", LEN(add_3_output), LEN(add_3_output[0]), LEN(add_3_output[0][0]));
}

void conv2d_6(){
    int conv2d_6_kernel[conv2d_6_kernel_size][conv2d_6_kernel_size][conv2d_6_kernel_channels][conv2d_6_kernel_num];
    int conv2d_6_bias[conv2d_6_kernel_num];
    for(int k = 0; k < conv2d_6_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for(int i = 0; i < conv2d_6_kernel_size; i++){
            for(int j = 0; j < conv2d_6_kernel_size; j++){
                for(int l = 0; l < conv2d_6_kernel_channels; l++){
                    conv2d_6_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_6_kernel[i][j][l][k]);
                }
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for(int k = 0; k < conv2d_6_kernel_num; k++){
        conv2d_6_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_6_bias[k]);
    }

    // Perform conv2d_6
    for(int k = 0; k < conv2d_6_kernel_num; k++){
        for(int i = 0; i < add_3_output_size; i++){
            for(int j = 0; j < add_3_output_size; j++){
                conv2d_6_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_6_kernel_channels; l++){
                    for(int m = 0; m < conv2d_6_kernel_size; m++){
                        for(int n = 0; n < conv2d_6_kernel_size; n++){
                            conv2d_6_output[i][j][k] += add_3_output[i*conv2d_6_kernel_stride + m][j*conv2d_6_kernel_stride + n][l] * conv2d_6_kernel[m][n][l][k];
                        }
                    }
                    conv2d_6_output[i][j][k] += conv2d_6_bias[k];
                }
            }
        }
    }

    printf("Size of conv2d_6_output: %d x %d x %d \n", LEN(conv2d_6_output), LEN(conv2d_6_output[0]), LEN(conv2d_6_output[0][0]));
}

void depthwiseconv2d_6(){
    int depthwiseconv2d_6_kernel[depthwiseconv2d_6_kernel_size][depthwiseconv2d_6_kernel_size][depthwiseconv2d_6_kernel_channels];
    int depthwiseconv2d_6_bias[depthwiseconv2d_6_kernel_channels];
    for(int k = 0; k < depthwiseconv2d_6_kernel_channels; k++){
        // printf("Kernel %d\n", k);
        for(int i = 0; i < depthwiseconv2d_6_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_6_kernel_size; j++){
                depthwiseconv2d_6_kernel[i][j][k] = rand() % 5 - 2;
                // printf("%f ", depthwiseconv2d_6_kernel[i][j][k]);
            }
            // printf("\n");
        }
    }

    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_6_kernel_channels; k++){
        depthwiseconv2d_6_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_6_bias[k]);
    }

    //Determine the padded conv2d_6_output size
    int conv2d_6_output_padded_size = conv2d_6_output_size + 2*depthwiseconv2d_6_kernel_padding;
    int conv2d_6_output_padded[conv2d_6_output_padded_size][conv2d_6_output_padded_size][depthwiseconv2d_6_kernel_channels];
    //Initialize the padded conv2d_6_output with 0
    for(int i = 0; i < conv2d_6_output_padded_size; i++){
        for(int j = 0; j < conv2d_6_output_padded_size; j++){
            for(int k = 0; k < depthwiseconv2d_6_kernel_channels; k++){
                conv2d_6_output_padded[i][j][k] = 0;
            }
        }
    }

    //Copy the conv2d_6_output to the padded conv2d_6_output
    for(int i = 1; i < conv2d_6_output_size - 1; i++){
        for(int j = 1; j < conv2d_6_output_size - 1; j++){
            for(int k = 1; k < depthwiseconv2d_6_kernel_channels; k++){
                conv2d_6_output_padded[i][j][k] = conv2d_6_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_6_output_padded: %d x %d x %d \n", LEN(conv2d_6_output_padded), LEN(conv2d_6_output_padded[0]), LEN(conv2d_6_output_padded[0][0]));

    //Perform depthwiseconv2d_6
    for(int k = 0; k < depthwiseconv2d_6_kernel_channels; k++){
        for(int i = 0; i < conv2d_6_output_size; i++){
            for(int j = 0; j < conv2d_6_output_size; j++){
                depthwiseconv2d_6_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_6_kernel_size; l++){
                    for(int m = 0; m < depthwiseconv2d_6_kernel_size; m++){
                        depthwiseconv2d_6_output[i][j][k] += conv2d_6_output_padded[i*depthwiseconv2d_6_kernel_stride + l][j*depthwiseconv2d_6_kernel_stride + m][k] * depthwiseconv2d_6_kernel[l][m][k];
                    }
                }
                depthwiseconv2d_6_output[i][j][k] += depthwiseconv2d_6_bias[k];
            }
        }
    }
    printf("Size of depthwiseconv2d_6_output: %d x %d x %d \n", LEN(depthwiseconv2d_6_output), LEN(depthwiseconv2d_6_output[0]), LEN(depthwiseconv2d_6_output[0][0]));
}

void pointwiseconv2d_6(){
    int pointwiseconv2d_6_kernel[pointwiseconv2d_6_kernel_size][pointwiseconv2d_6_kernel_size][pointwiseconv2d_6_kernel_channels][pointwiseconv2d_6_kernel_num];
    int pointwiseconv2d_6_bias[pointwiseconv2d_6_kernel_channels];
    for (int k = 0; k < pointwiseconv2d_6_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < pointwiseconv2d_6_kernel_size; i++){
            for(int j = 0; j < pointwiseconv2d_6_kernel_size; j++){
                for(int l = 0; l < pointwiseconv2d_6_kernel_channels; l++){
                    pointwiseconv2d_6_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_6_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_6_kernel_num; k++){
        pointwiseconv2d_6_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_6_bias[k]);
    }

    //Perform pointwiseconv2d_6
    for(int k = 0; k < pointwiseconv2d_6_kernel_num; k++){
        for(int i = 0; i < conv2d_6_output_size; i++){
            for(int j = 0; j < conv2d_6_output_size; j++){
                pointwiseconv2d_6_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_6_kernel_channels; l++){
                    for(int m = 0; m < pointwiseconv2d_6_kernel_size; m++){
                        for(int n = 0; n < pointwiseconv2d_6_kernel_size; n++){
                            pointwiseconv2d_6_output[i][j][k] += depthwiseconv2d_6_output[i*pointwiseconv2d_6_kernel_stride + m][j*pointwiseconv2d_6_kernel_stride + n][l] * pointwiseconv2d_6_kernel[m][n][l][k];
                        }
                    }
                }
                pointwiseconv2d_6_output[i][j][k] += pointwiseconv2d_6_bias[k];
            }
        }
    }
    printf("Size of pointwiseconv2d_6_output: %d x %d x %d \n", LEN(pointwiseconv2d_6_output), LEN(pointwiseconv2d_6_output[0]), LEN(pointwiseconv2d_6_output[0][0]));
}

void add_4(){
    for(int i = 0; i < conv2d_6_output_size; i++){
        for(int j = 0; j < conv2d_6_output_size; j++){
            for(int k = 0; k < pointwiseconv2d_6_kernel_num; k++){
                add_4_output[i][j][k] = pointwiseconv2d_6_output[i][j][k] + add_3_output[i][j][k];
            }
        }
    }
    printf("Size of add_4_output: %d x %d x %d \n", LEN(add_4_output), LEN(add_4_output[0]), LEN(add_4_output[0][0]));
}

void conv2d_7(){
    int conv2d_7_kernel[conv2d_7_kernel_size][conv2d_7_kernel_size][conv2d_7_kernel_channels][conv2d_7_kernel_num];
    int conv2d_7_bias[conv2d_7_kernel_num];
    for (int k = 0; k < conv2d_7_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_7_kernel_size; i++){
            for(int j = 0; j < conv2d_7_kernel_size; j++){
                for(int l = 0; l < conv2d_7_kernel_channels; l++){
                    conv2d_7_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_7_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_7_kernel_num; k++){
        conv2d_7_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_7_bias[k]);
    }

    //Perform conv2d_7
    for(int k = 0; k < conv2d_7_kernel_num; k++){
        for(int i = 0; i < add_4_output_size; i++){
            for(int j = 0; j < add_4_output_size; j++){
                conv2d_7_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_7_kernel_channels; l++){
                    for(int m = 0; m < conv2d_7_kernel_size; m++){
                        for(int n = 0; n < conv2d_7_kernel_size; n++){
                            conv2d_7_output[i][j][k] += add_4_output[i*conv2d_7_kernel_stride + m][j*conv2d_7_kernel_stride + n][l] * conv2d_7_kernel[m][n][l][k];
                        }
                    }
                }
                conv2d_7_output[i][j][k] += conv2d_7_bias[k];
            }
        }
    }

    printf("Size of conv2d_7_output: %d x %d x %d \n", LEN(conv2d_7_output), LEN(conv2d_7_output[0]), LEN(conv2d_7_output[0][0]));
}

void depthwiseconv2d_7(){
    int depthwiseconv2d_7_kernel[depthwiseconv2d_7_kernel_size][depthwiseconv2d_7_kernel_size][depthwiseconv2d_7_kernel_channels][depthwiseconv2d_7_kernel_num];
    int depthwiseconv2d_7_bias[depthwiseconv2d_7_kernel_num];
    for (int k = 0; k < depthwiseconv2d_7_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < depthwiseconv2d_7_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_7_kernel_size; j++){
                for(int l = 0; l < depthwiseconv2d_7_kernel_channels; l++){
                    depthwiseconv2d_7_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", depthwiseconv2d_7_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_7_kernel_channels; k++){
        depthwiseconv2d_7_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_7_bias[k]);
    }

    //Determine the padded conv2d_7_output size
    int conv2d_7_output_padded_size = conv2d_7_output_size + 2*depthwiseconv2d_7_kernel_padding;
    int conv2d_7_output_padded[conv2d_7_output_padded_size][conv2d_7_output_padded_size][depthwiseconv2d_7_kernel_channels];

    //Initialize padded conv2d_7_output with 0
    for(int i = 0; i < conv2d_7_output_padded_size; i++){
        for(int j = 0; j < conv2d_7_output_padded_size; j++){
            for(int k = 0; k < depthwiseconv2d_7_kernel_channels; k++){
                conv2d_7_output_padded[i][j][k] = 0;
            }
        }
    }

    //Copy conv2d_7_output to padded conv2d_7_output
    for(int i = 0; i < conv2d_7_output_size; i++){
        for(int j = 0; j < conv2d_7_output_size; j++){
            for(int k = 0; k < depthwiseconv2d_7_kernel_channels; k++){
                conv2d_7_output_padded[i][j][k] = conv2d_7_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_7_output_padded: %d x %d x %d \n", LEN(conv2d_7_output_padded), LEN(conv2d_7_output_padded[0]), LEN(conv2d_7_output_padded[0][0]));

    //Perform depthwiseconv2d_7
    for(int k = 0; k < depthwiseconv2d_7_kernel_channels; k++){
        for(int i = 0; i < depthwiseconv2d_7_output_size; i++){
            for(int j = 0; j < depthwiseconv2d_7_output_size; j++){
                depthwiseconv2d_7_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_7_kernel_size; l++){
                    for(int m = 0; m < depthwiseconv2d_7_kernel_size; m++){
                        depthwiseconv2d_7_output[i][j][k] += conv2d_7_output_padded[i*depthwiseconv2d_7_kernel_stride + l][j*depthwiseconv2d_7_kernel_stride + m][k] * depthwiseconv2d_7_kernel[l][m][k][k];
                    }
                }
                depthwiseconv2d_7_output[i][j][k] += depthwiseconv2d_7_bias[k];
            }
        }
    }

    printf("Size of depthwiseconv2d_7_output: %d x %d x %d \n", LEN(depthwiseconv2d_7_output), LEN(depthwiseconv2d_7_output[0]), LEN(depthwiseconv2d_7_output[0][0]));
}

void pointwiseconv2d_7(){
    int pointwiseconv2d_7_kernel[pointwiseconv2d_7_kernel_size][pointwiseconv2d_7_kernel_size][pointwiseconv2d_7_kernel_channels][pointwiseconv2d_7_kernel_num];
    int pointwiseconv2d_7_bias[pointwiseconv2d_7_kernel_num];
    for (int k = 0; k < pointwiseconv2d_7_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < pointwiseconv2d_7_kernel_size; i++){
            for(int j = 0; j < pointwiseconv2d_7_kernel_size; j++){
                for(int l = 0; l < pointwiseconv2d_7_kernel_channels; l++){
                    pointwiseconv2d_7_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_7_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_7_kernel_num; k++){
        pointwiseconv2d_7_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_7_bias[k]);
    }

    //Perform pointwiseconv2d_7
    for(int k = 0; k < pointwiseconv2d_7_kernel_num; k++){
        for(int i = 0; i < pointwiseconv2d_7_output_size; i++){
            for(int j = 0; j < pointwiseconv2d_7_output_size; j++){
                pointwiseconv2d_7_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_7_kernel_channels; l++){
                    for(int m = 0; m < pointwiseconv2d_7_kernel_size; m++){
                        for(int n = 0; n < pointwiseconv2d_7_kernel_size; n++){
                            pointwiseconv2d_7_output[i][j][k] += depthwiseconv2d_7_output[i][j][l] * pointwiseconv2d_7_kernel[m][n][l][k];
                        }
                    }
                }
                pointwiseconv2d_7_output[i][j][k] += pointwiseconv2d_7_bias[k];
            }
        }
    }
    printf("Size of pointwiseconv2d_7_output: %d x %d x %d \n", LEN(pointwiseconv2d_7_output), LEN(pointwiseconv2d_7_output[0]), LEN(pointwiseconv2d_7_output[0][0]));
}

int main (){
    init_input();
    conv2d_1();
    depthwiseconv2d_1();
    pointwiseconv2d_1();
    add_1();
    conv2d_2();
    depthwiseconv2d_2();
    pointwiseconv2d_2();
    conv2d_3();
    depthwiseconv2d_3();
    pointwiseconv2d_3();
    add_2();
    conv2d_4();
    depthwiseconv2d_4();
    pointwiseconv2d_4();
    conv2d_5();
    depthwiseconv2d_5();
    pointwiseconv2d_5();
    add_3();
    conv2d_6();
    depthwiseconv2d_6();
    pointwiseconv2d_6();
    add_4();
    conv2d_7();
    depthwiseconv2d_7();
    pointwiseconv2d_7();
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "weights/weights.h"

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
#define conv2d_1_output_size 112
int conv2d_1_output[112][112][16];

//DepthwiseConv2d_1 112x112x16
#define depthwiseconv2d_1_kernel_size 3
#define depthwiseconv2d_1_kernel_channels 16
#define depthwiseconv2d_1_kernel_num 1
#define depthwiseconv2d_1_kernel_stride 1
#define depthwiseconv2d_1_kernel_padding 1
#define depthwiseconv2d_1_output_size 112
int depthwiseconv2d_1_output[112][112][16];

//PointwiseConv2d_1 112x112x16
#define pointwiseconv2d_1_kernel_size 1
#define pointwiseconv2d_1_kernel_channels 16
#define pointwiseconv2d_1_kernel_num 16
#define pointwiseconv2d_1_kernel_stride 1
#define pointwiseconv2d_1_kernel_padding 0
#define pointwiseconv2d_1_output_size 112
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
#define conv2d_2_output_size 112
int conv2d_2_output[112][112][64];

//DepthwiseConv2d_2 56x56x64
#define depthwiseconv2d_2_kernel_size 3
#define depthwiseconv2d_2_kernel_channels 64
#define depthwiseconv2d_2_kernel_num 1
#define depthwiseconv2d_2_kernel_stride 2
#define depthwiseconv2d_2_kernel_padding (1.0/2.0)
#define depthwiseconv2d_2_output_size 56
int depthwiseconv2d_2_output[56][56][64];

//PointwiseConv2d_2 56x56x24
#define pointwiseconv2d_2_kernel_size 1
#define pointwiseconv2d_2_kernel_channels 64
#define pointwiseconv2d_2_kernel_num 24
#define pointwiseconv2d_2_kernel_stride 1
#define pointwiseconv2d_2_kernel_padding 0
#define pointwiseconv2d_2_output_size 56
int pointwiseconv2d_2_output[56][56][24];

//Conv2d_3 56x56x72
#define conv2d_3_kernel_size 1
#define conv2d_3_kernel_channels 24
#define conv2d_3_kernel_num 72
#define conv2d_3_kernel_stride 1
#define conv2d_3_kernel_padding 0
#define conv2d_3_output_size 56
int conv2d_3_output[56][56][72];

//DepthwiseConv2d_3 56x56x72
#define depthwiseconv2d_3_kernel_size 3
#define depthwiseconv2d_3_kernel_channels 72
#define depthwiseconv2d_3_kernel_num 1
#define depthwiseconv2d_3_kernel_stride 1
#define depthwiseconv2d_3_kernel_padding 1
#define depthwiseconv2d_3_output_size 56
int depthwiseconv2d_3_output[56][56][72];

//PointwiseConv2d_3 56x56x24
#define pointwiseconv2d_3_kernel_size 1
#define pointwiseconv2d_3_kernel_channels 72
#define pointwiseconv2d_3_kernel_num 24
#define pointwiseconv2d_3_kernel_stride 1
#define pointwiseconv2d_3_kernel_padding 0
#define pointwiseconv2d_3_output_size 56
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
#define conv2d_4_output_size 56
int conv2d_4_output[56][56][72];

//DepthwiseConv2d_4 28x28x72
#define depthwiseconv2d_4_kernel_size 3
#define depthwiseconv2d_4_kernel_channels 72
#define depthwiseconv2d_4_kernel_num 1
#define depthwiseconv2d_4_kernel_stride 2
#define depthwiseconv2d_4_kernel_padding (1.0/2.0)
#define depthwiseconv2d_4_output_size 28
int depthwiseconv2d_4_output[28][28][72];

//PointwiseConv2d_4 28x28x40
#define pointwiseconv2d_4_kernel_size 1
#define pointwiseconv2d_4_kernel_channels 72
#define pointwiseconv2d_4_kernel_num 40
#define pointwiseconv2d_4_kernel_stride 1
#define pointwiseconv2d_4_kernel_padding 0
#define pointwiseconv2d_4_output_size 28
int pointwiseconv2d_4_output[28][28][40];

//Conv2d_5 28x28x120
#define conv2d_5_kernel_size 1
#define conv2d_5_kernel_channels 40
#define conv2d_5_kernel_num 120
#define conv2d_5_kernel_stride 1
#define conv2d_5_kernel_padding 0
#define conv2d_5_output_size 28
int conv2d_5_output[28][28][120];

//DepthwiseConv2d_5 28x28x120
#define depthwiseconv2d_5_kernel_size 3
#define depthwiseconv2d_5_kernel_channels 120
#define depthwiseconv2d_5_kernel_num 1
#define depthwiseconv2d_5_kernel_stride 1
#define depthwiseconv2d_5_kernel_padding 1
#define depthwiseconv2d_5_output_size 28
int depthwiseconv2d_5_output[28][28][120];

//PointwiseConv2d_5 28x28x40
#define pointwiseconv2d_5_kernel_size 1
#define pointwiseconv2d_5_kernel_channels 120
#define pointwiseconv2d_5_kernel_num 40
#define pointwiseconv2d_5_kernel_stride 1
#define pointwiseconv2d_5_kernel_padding 0
#define pointwiseconv2d_5_output_size 28
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
#define conv2d_6_output_size 28
int conv2d_6_output[28][28][120];

//DepthwiseConv2d_6 28x28x120
#define depthwiseconv2d_6_kernel_size 3
#define depthwiseconv2d_6_kernel_channels 120
#define depthwiseconv2d_6_kernel_num 1
#define depthwiseconv2d_6_kernel_stride 1
#define depthwiseconv2d_6_kernel_padding 1
#define depthwiseconv2d_6_output_size 28
int depthwiseconv2d_6_output[28][28][120];

//PointwiseConv2d_6 28x28x40
#define pointwiseconv2d_6_kernel_size 1
#define pointwiseconv2d_6_kernel_channels 120
#define pointwiseconv2d_6_kernel_num 40
#define pointwiseconv2d_6_kernel_stride 1
#define pointwiseconv2d_6_kernel_padding 0
#define pointwiseconv2d_6_output_size 28
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
#define conv2d_7_output_size 28
int conv2d_7_output[28][28][240];

//DepthwiseConv2d_7 14x14x240
#define depthwiseconv2d_7_kernel_size 3
#define depthwiseconv2d_7_kernel_channels 240
#define depthwiseconv2d_7_kernel_num 1
#define depthwiseconv2d_7_kernel_stride 2
#define depthwiseconv2d_7_kernel_padding (1.0/2.0)
#define depthwiseconv2d_7_output_size 14
int depthwiseconv2d_7_output[14][14][240];

//PointwiseConv2d_7 14x14x80
#define pointwiseconv2d_7_kernel_size 1
#define pointwiseconv2d_7_kernel_channels 240
#define pointwiseconv2d_7_kernel_num 80
#define pointwiseconv2d_7_kernel_stride 1
#define pointwiseconv2d_7_kernel_padding 0
#define pointwiseconv2d_7_output_size 14
int pointwiseconv2d_7_output[14][14][80];

// START - BIG NOTICE - MISSED

//Conv2d_8_bonus 14x14x200
#define conv2d_8_bonus_kernel_size 1
#define conv2d_8_bonus_kernel_channels 80
#define conv2d_8_bonus_kernel_num 200
#define conv2d_8_bonus_kernel_stride 1
#define conv2d_8_bonus_kernel_padding 0
#define conv2d_8_bonus_output_size 14
int conv2d_8_bonus_output[14][14][200];

//DepthwiseConv2d_8_bonus 14x14x200
#define depthwiseconv2d_8_bonus_kernel_size 3
#define depthwiseconv2d_8_bonus_kernel_channels 200
#define depthwiseconv2d_8_bonus_kernel_num 1
#define depthwiseconv2d_8_bonus_kernel_stride 1
#define depthwiseconv2d_8_bonus_kernel_padding 1
#define depthwiseconv2d_8_bonus_output_size 14
int depthwiseconv2d_8_bonus_output[14][14][200];

//PointwiseConv2d_8_bonus 14x14x80
#define pointwiseconv2d_8_bonus_kernel_size 1
#define pointwiseconv2d_8_bonus_kernel_channels 200
#define pointwiseconv2d_8_bonus_kernel_num 80
#define pointwiseconv2d_8_bonus_kernel_stride 1
#define pointwiseconv2d_8_bonus_kernel_padding 0
#define pointwiseconv2d_8_bonus_output_size 14
int pointwiseconv2d_8_bonus_output[14][14][80];

// STOP - BIG NOTICE - MISSED

//Add_5 14x14x80
#define add_5_output_size 14
#define add_5_output_channels 80
int add_5_output[14][14][80];

//Conv2d_8 14x14x184
#define conv2d_8_kernel_size 1
#define conv2d_8_kernel_channels 80
#define conv2d_8_kernel_num 200
#define conv2d_8_kernel_stride 1
#define conv2d_8_kernel_padding 0
#define conv2d_8_output_size 14
int conv2d_8_output[14][14][200];

//DepthwiseConv2d_8 14x14x184
#define depthwiseconv2d_8_kernel_size 3
#define depthwiseconv2d_8_kernel_channels 200
#define depthwiseconv2d_8_kernel_num 1
#define depthwiseconv2d_8_kernel_stride 1
#define depthwiseconv2d_8_kernel_padding 1
#define depthwiseconv2d_8_output_size 14
int depthwiseconv2d_8_output[14][14][200];

//PointwiseConv2d_8 14x14x80
#define pointwiseconv2d_8_kernel_size 1
#define pointwiseconv2d_8_kernel_channels 200
#define pointwiseconv2d_8_kernel_num 80
#define pointwiseconv2d_8_kernel_stride 1
#define pointwiseconv2d_8_kernel_padding 0
#define pointwiseconv2d_8_output_size 14
int pointwiseconv2d_8_output[14][14][80];

//Add_6 14x14x80
#define add_6_output_size 14
#define add_6_output_channels 80
int add_6_output[14][14][80];

//Conv2d_9 14x14x184
#define conv2d_9_kernel_size 1
#define conv2d_9_kernel_channels 80
#define conv2d_9_kernel_num 184
#define conv2d_9_kernel_stride 1
#define conv2d_9_kernel_padding 0
#define conv2d_9_output_size 14
int conv2d_9_output[14][14][184];

//DepthwiseConv2d_9 14x14x184
#define depthwiseconv2d_9_kernel_size 3
#define depthwiseconv2d_9_kernel_channels 184
#define depthwiseconv2d_9_kernel_num 1
#define depthwiseconv2d_9_kernel_stride 1
#define depthwiseconv2d_9_kernel_padding 1
#define depthwiseconv2d_9_output_size 14
int depthwiseconv2d_9_output[14][14][184];

//PointwiseConv2d_9 14x14x80
#define pointwiseconv2d_9_kernel_size 1
#define pointwiseconv2d_9_kernel_channels 184
#define pointwiseconv2d_9_kernel_num 80
#define pointwiseconv2d_9_kernel_stride 1
#define pointwiseconv2d_9_kernel_padding 0
#define pointwiseconv2d_9_output_size 14
int pointwiseconv2d_9_output[14][14][80];

//Add_7 14x14x80
#define add_7_output_size 14
#define add_7_output_channels 80
int add_7_output[14][14][80];

//Conv2d_10 14x14x480
#define conv2d_10_kernel_size 1
#define conv2d_10_kernel_channels 80
#define conv2d_10_kernel_num 480
#define conv2d_10_kernel_stride 1
#define conv2d_10_kernel_padding 0
#define conv2d_10_output_size 14
int conv2d_10_output[14][14][480];

//DepthwiseConv2d_10 14x14x480
#define depthwiseconv2d_10_kernel_size 3
#define depthwiseconv2d_10_kernel_channels 480
#define depthwiseconv2d_10_kernel_num 1
#define depthwiseconv2d_10_kernel_stride 1
#define depthwiseconv2d_10_kernel_padding 1
#define depthwiseconv2d_10_output_size 14
int depthwiseconv2d_10_output[14][14][480];

//PointwiseConv2d_10 14x14x80
#define pointwiseconv2d_10_kernel_size 1
#define pointwiseconv2d_10_kernel_channels 480
#define pointwiseconv2d_10_kernel_num 112
#define pointwiseconv2d_10_kernel_stride 1
#define pointwiseconv2d_10_kernel_padding 0
#define pointwiseconv2d_10_output_size 14
int pointwiseconv2d_10_output[14][14][112];

//Conv2d_11 14x14x672
#define conv2d_11_kernel_size 1
#define conv2d_11_kernel_channels 112
#define conv2d_11_kernel_num 672
#define conv2d_11_kernel_stride 1
#define conv2d_11_kernel_padding 0
#define conv2d_11_output_size 14
int conv2d_11_output[14][14][672];

//DepthwiseConv2d_11 14x14x672
#define depthwiseconv2d_11_kernel_size 3
#define depthwiseconv2d_11_kernel_channels 672
#define depthwiseconv2d_11_kernel_num 1
#define depthwiseconv2d_11_kernel_stride 1
#define depthwiseconv2d_11_kernel_padding 1
#define depthwiseconv2d_11_output_size 14
int depthwiseconv2d_11_output[14][14][672];

//PointwiseConv2d_11 14x14x80
#define pointwiseconv2d_11_kernel_size 1
#define pointwiseconv2d_11_kernel_channels 672
#define pointwiseconv2d_11_kernel_num 112
#define pointwiseconv2d_11_kernel_stride 1
#define pointwiseconv2d_11_kernel_padding 0
#define pointwiseconv2d_11_output_size 14
int pointwiseconv2d_11_output[14][14][112];

//Add_8 14x14x112
#define add_8_output_size 14
#define add_8_output_channels 112
int add_8_output[14][14][112];

//Conv2d_12 14x14x672
#define conv2d_12_kernel_size 1
#define conv2d_12_kernel_channels 112
#define conv2d_12_kernel_num 672
#define conv2d_12_kernel_stride 1
#define conv2d_12_kernel_padding 0
#define conv2d_12_output_size 14
int conv2d_12_output[14][14][672];

//DepthwiseConv2d_12 7x7x672
#define depthwiseconv2d_12_kernel_size 3
#define depthwiseconv2d_12_kernel_channels 672
#define depthwiseconv2d_12_kernel_num 1
#define depthwiseconv2d_12_kernel_stride 2
#define depthwiseconv2d_12_kernel_padding (1.0/2.0)
#define depthwiseconv2d_12_output_size 7
int depthwiseconv2d_12_output[7][7][672];

//PointwiseConv2d_12 7x7x160
#define pointwiseconv2d_12_kernel_size 1
#define pointwiseconv2d_12_kernel_channels 672
#define pointwiseconv2d_12_kernel_num 160
#define pointwiseconv2d_12_kernel_stride 1
#define pointwiseconv2d_12_kernel_padding 0
#define pointwiseconv2d_12_output_size 7
int pointwiseconv2d_12_output[7][7][160];

//Conv2d_13 7x7x960
#define conv2d_13_kernel_size 1
#define conv2d_13_kernel_channels 160
#define conv2d_13_kernel_num 960
#define conv2d_13_kernel_stride 1
#define conv2d_13_kernel_padding 0
#define conv2d_13_output_size 7
int conv2d_13_output[7][7][960];

//DepthwiseConv2d_13 7x7x960
#define depthwiseconv2d_13_kernel_size 3
#define depthwiseconv2d_13_kernel_channels 960
#define depthwiseconv2d_13_kernel_num 1
#define depthwiseconv2d_13_kernel_stride 1
#define depthwiseconv2d_13_kernel_padding 1
#define depthwiseconv2d_13_output_size 7
int depthwiseconv2d_13_output[7][7][960];

//PointwiseConv2d_13 7x7x160
#define pointwiseconv2d_13_kernel_size 1
#define pointwiseconv2d_13_kernel_channels 960
#define pointwiseconv2d_13_kernel_num 160
#define pointwiseconv2d_13_kernel_stride 1
#define pointwiseconv2d_13_kernel_padding 0
#define pointwiseconv2d_13_output_size 7
int pointwiseconv2d_13_output[7][7][160];

//Add_9 7x7x160
#define add_9_output_size 7
#define add_9_output_channels 160
int add_9_output[7][7][160];

//Conv2d_14 7x7x960
#define conv2d_14_kernel_size 1
#define conv2d_14_kernel_channels 160
#define conv2d_14_kernel_num 960
#define conv2d_14_kernel_stride 1
#define conv2d_14_kernel_padding 0
#define conv2d_14_output_size 7
int conv2d_14_output[7][7][960];

//DepthwiseConv2d_14 7x7x960
#define depthwiseconv2d_14_kernel_size 3
#define depthwiseconv2d_14_kernel_channels 960
#define depthwiseconv2d_14_kernel_num 1
#define depthwiseconv2d_14_kernel_stride 1
#define depthwiseconv2d_14_kernel_padding 1
#define depthwiseconv2d_14_output_size 7
int depthwiseconv2d_14_output[7][7][960];

//PointwiseConv2d_14 7x7x160
#define pointwiseconv2d_14_kernel_size 1
#define pointwiseconv2d_14_kernel_channels 960
#define pointwiseconv2d_14_kernel_num 160
#define pointwiseconv2d_14_kernel_stride 1
#define pointwiseconv2d_14_kernel_padding 0
#define pointwiseconv2d_14_output_size 7
int pointwiseconv2d_14_output[7][7][160];

//Add_10 7x7x160
#define add_10_output_size 7
#define add_10_output_channels 160
int add_10_output[7][7][160];

//Conv2d_15 7x7x960
#define conv2d_15_kernel_size 1
#define conv2d_15_kernel_channels 160
#define conv2d_15_kernel_num 960
#define conv2d_15_kernel_stride 1
#define conv2d_15_kernel_padding 0
#define conv2d_15_output_size 7
int conv2d_15_output[7][7][960];

//AveragePool2d_1 1x1x960
#define averagepool2d_1_kernel_size 7
#define averagepool2d_1_kernel_stride 1
#define averagepool2d_1_kernel_padding 0
#define averagepool2d_1_output_size 1
int averagepool2d_1_output[1][1][960];

//Conv2d_16 1x1x1280
#define conv2d_16_kernel_size 1
#define conv2d_16_kernel_channels 960
#define conv2d_16_kernel_num 1280
#define conv2d_16_kernel_stride 1
#define conv2d_16_kernel_padding 0
#define conv2d_16_output_size 1
int conv2d_16_output[1][1][1280];

//AveragePool2d_2 1x1x1280
#define averagepool2d_2_kernel_size 1
#define averagepool2d_2_kernel_stride 1
#define averagepool2d_2_kernel_padding 0
#define averagepool2d_2_output_size 1
int averagepool2d_2_output[1][1][1280];

//Conv2d_17 1x1x1001
#define conv2d_17_kernel_size 1
#define conv2d_17_kernel_channels 1280
#define conv2d_17_kernel_num 1001
#define conv2d_17_kernel_stride 1
#define conv2d_17_kernel_padding 0
#define conv2d_17_output_size 1
int conv2d_17_output[1][1][1001];

//Reshape 1001
int reshape_output[1001];

//Softmax 1001
int softmax_output[1001];

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
    printf("Size of input: %d x %d x %d \n", LEN(input), LEN(input[0]), LEN(input[0][0]));
}

void conv2d_1() {
    static int conv2d_1_kernel[conv2d_1_kernel_size][conv2d_1_kernel_size][conv2d_1_kernel_channels][conv2d_1_kernel_num];
    static int conv2d_1_bias[conv2d_1_kernel_num];
    for (int k = 0; k < conv2d_1_kernel_num; k++){
        // printf("Kernel %d\n", k);
        for (int l = 0; l < conv2d_1_kernel_channels; l++){
            for (int i = 0; i < conv2d_1_kernel_size; i++) {
                for (int j = 0; j < conv2d_1_kernel_size; j++) {
                    conv2d_1_kernel[i][j][l][k] = conv2d_1_weights[i][j][l][k];
                    // printf("%f ", conv2d_1_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
        }
    }


    for (int k = 0; k < conv2d_1_kernel_num; k++){
        conv2d_1_bias[k] = conv2d_1_biases[k];
        // printf("%f ", conv2d_1_bias[k]);
    }
    // Copy input to padded input
    static const int padded_size = 225;
    static int input_padded[225][225][input_channels];

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
    printf("Size of padded input: %d x %d x %d \n", LEN(input_padded), LEN(input_padded[0]), LEN(input_padded[0][0]));

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
}

void depthwiseconv2d_1(){
    static int depthwiseconv2d_1_kernel[depthwiseconv2d_1_kernel_size][depthwiseconv2d_1_kernel_size][depthwiseconv2d_1_kernel_channels][depthwiseconv2d_1_kernel_num];
    static int depthwiseconv2d_1_bias[depthwiseconv2d_1_kernel_channels];
    for (int k = 0; k < depthwiseconv2d_1_kernel_num; k++){
        for (int l = 0; l < depthwiseconv2d_1_kernel_channels; l++){
            for (int i = 0; i < depthwiseconv2d_1_kernel_size; i++) {
                for (int j = 0; j < depthwiseconv2d_1_kernel_size; j++) {
                    depthwiseconv2d_1_kernel[i][j][l][k] = depthwiseconv2d_1_weights[i][j][l][k];
                        // printf("%f ", depthwiseconv2d_1_kernel[i][j][l][k]);  
                    }
                // printf("\n");
                }
            }
    }
    for (int k = 0; k < depthwiseconv2d_1_kernel_channels; k++){
        depthwiseconv2d_1_bias[k] = depthwiseconv2d_1_biases[k];
        // printf("%f ", depthwiseconv2d_1_bias[k]);  
    }

    // Initialize padded conv2d_1_output as 0 array
    static const int conv2d_1_output_padded_size = 114;
    static int conv2d_1_output_padded[114][114][conv2d_1_kernel_num];
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
    static int pointwiseconv2d_1_kernel[pointwiseconv2d_1_kernel_size][pointwiseconv2d_1_kernel_size][pointwiseconv2d_1_kernel_channels][pointwiseconv2d_1_kernel_num];
    static int pointwiseconv2d_1_bias[pointwiseconv2d_1_kernel_num];
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
    static int conv2d_2_kernel[conv2d_2_kernel_size][conv2d_2_kernel_size][conv2d_2_kernel_channels][conv2d_2_kernel_num];
    static int conv2d_2_bias[conv2d_2_kernel_num];
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
    static int depthwiseconv2d_2_kernel[depthwiseconv2d_2_kernel_size][depthwiseconv2d_2_kernel_size][depthwiseconv2d_2_kernel_channels][depthwiseconv2d_2_kernel_num];
    static int depthwiseconv2d_2_bias[depthwiseconv2d_2_kernel_channels];
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
    static const int conv2d_2_output_padded_size = 113;
    static int conv2d_2_output_padded[113][113][conv2d_2_kernel_num];
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
    static int pointwiseconv2d_2_kernel[pointwiseconv2d_2_kernel_size][pointwiseconv2d_2_kernel_size][pointwiseconv2d_2_kernel_channels][pointwiseconv2d_2_kernel_num];
    static int pointwiseconv2d_2_bias[pointwiseconv2d_2_kernel_num];
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
    static int conv2d_3_kernel[conv2d_3_kernel_size][conv2d_3_kernel_size][conv2d_3_kernel_channels][conv2d_3_kernel_num];
    static int conv2d_3_bias[conv2d_3_kernel_num];
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
    static int depthwiseconv2d_3_kernel[depthwiseconv2d_3_kernel_size][depthwiseconv2d_3_kernel_size][depthwiseconv2d_3_kernel_channels];
    static int depthwiseconv2d_3_bias[depthwiseconv2d_3_kernel_channels];
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
    static const int padded_conv2d_3_output_size = 58;
    static int padded_conv2d_3_output[58][58][conv2d_3_kernel_num];
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
    static int pointwiseconv2d_3_kernel[pointwiseconv2d_3_kernel_size][pointwiseconv2d_3_kernel_size][pointwiseconv2d_3_kernel_channels][pointwiseconv2d_3_kernel_num];
    static int pointwiseconv2d_3_bias[pointwiseconv2d_3_kernel_num];
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
    static int conv2d_4_kernel[conv2d_4_kernel_size][conv2d_4_kernel_size][conv2d_4_kernel_channels][conv2d_4_kernel_num];
    static int conv2d_4_bias[conv2d_4_kernel_num];
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
    static int depthwiseconv2d_4_kernel[depthwiseconv2d_4_kernel_size][depthwiseconv2d_4_kernel_size][depthwiseconv2d_4_kernel_channels];
    static int depthwiseconv2d_4_bias[depthwiseconv2d_4_kernel_channels];
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
    static const int conv2d_4_output_padded_size = 57;
    static int conv2d_4_output_padded[57][57][conv2d_4_kernel_num];
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
    static int pointwiseconv2d_4_kernel[pointwiseconv2d_4_kernel_size][pointwiseconv2d_4_kernel_size][pointwiseconv2d_4_kernel_channels][pointwiseconv2d_4_kernel_num];
    static int pointwiseconv2d_4_bias[pointwiseconv2d_4_kernel_num];
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
    static int conv2d_5_kernel[conv2d_5_kernel_size][conv2d_5_kernel_size][conv2d_5_kernel_channels][conv2d_5_kernel_num];
    static int conv2d_5_bias[conv2d_5_kernel_num];
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
    static int depthwiseconv2d_5_kernel[depthwiseconv2d_5_kernel_size][depthwiseconv2d_5_kernel_size][depthwiseconv2d_5_kernel_channels][depthwiseconv2d_5_kernel_num];
    static int depthwiseconv2d_5_bias[depthwiseconv2d_5_kernel_channels];
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
    static const int conv2d_5_padded_output_size = 30;
    static int conv2d_5_padded_output[30][30][conv2d_5_kernel_num];
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
    static int pointwiseconv2d_5_kernel[pointwiseconv2d_5_kernel_size][pointwiseconv2d_5_kernel_size][pointwiseconv2d_5_kernel_channels][pointwiseconv2d_5_kernel_num];
    static int pointwiseconv2d_5_bias[pointwiseconv2d_5_kernel_num];
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
    static int conv2d_6_kernel[conv2d_6_kernel_size][conv2d_6_kernel_size][conv2d_6_kernel_channels][conv2d_6_kernel_num];
    static int conv2d_6_bias[conv2d_6_kernel_num];
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
    static int depthwiseconv2d_6_kernel[depthwiseconv2d_6_kernel_size][depthwiseconv2d_6_kernel_size][depthwiseconv2d_6_kernel_channels];
    static int depthwiseconv2d_6_bias[depthwiseconv2d_6_kernel_channels];
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
    static const int conv2d_6_output_padded_size = 30;
    static int conv2d_6_output_padded[30][30][depthwiseconv2d_6_kernel_channels];
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
    static int pointwiseconv2d_6_kernel[pointwiseconv2d_6_kernel_size][pointwiseconv2d_6_kernel_size][pointwiseconv2d_6_kernel_channels][pointwiseconv2d_6_kernel_num];
    static int pointwiseconv2d_6_bias[pointwiseconv2d_6_kernel_num];
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
    static int conv2d_7_kernel[conv2d_7_kernel_size][conv2d_7_kernel_size][conv2d_7_kernel_channels][conv2d_7_kernel_num];
    static int conv2d_7_bias[conv2d_7_kernel_num];
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
    static int depthwiseconv2d_7_kernel[depthwiseconv2d_7_kernel_size][depthwiseconv2d_7_kernel_size][depthwiseconv2d_7_kernel_channels][depthwiseconv2d_7_kernel_num];
    static int depthwiseconv2d_7_bias[depthwiseconv2d_7_kernel_channels];
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
    static const int conv2d_7_output_padded_size = 29;
    static int conv2d_7_output_padded[29][29][depthwiseconv2d_7_kernel_channels];

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
    static int pointwiseconv2d_7_kernel[pointwiseconv2d_7_kernel_size][pointwiseconv2d_7_kernel_size][pointwiseconv2d_7_kernel_channels][pointwiseconv2d_7_kernel_num];
    static int pointwiseconv2d_7_bias[pointwiseconv2d_7_kernel_num];
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

void conv2d_8_bonus(){
    static int conv2d_8_bonus_kernel[conv2d_8_bonus_kernel_size][conv2d_8_bonus_kernel_size][conv2d_8_bonus_kernel_channels][conv2d_8_bonus_kernel_num];
    static int conv2d_8_bonus_bias[conv2d_8_bonus_kernel_num];
    for (int k = 0; k < conv2d_8_bonus_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_8_bonus_kernel_size; i++){
            for(int j = 0; j < conv2d_8_bonus_kernel_size; j++){
                for(int l = 0; l < conv2d_8_bonus_kernel_channels; l++){
                    conv2d_8_bonus_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_8_bonus_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_8_bonus_kernel_num; k++){
        conv2d_8_bonus_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_8_bonus_bias[k]);
    }

    //Perform conv2d_8_bonus
    for(int k = 0; k < conv2d_8_bonus_kernel_num; k++){
        for(int i = 0; i < conv2d_8_bonus_output_size; i++){
            for(int j = 0; j < conv2d_8_bonus_output_size; j++){
                conv2d_8_bonus_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_8_bonus_kernel_channels; l++){
                    for(int m = 0; m < conv2d_8_bonus_kernel_size; m++){
                        for(int n = 0; n < conv2d_8_bonus_kernel_size; n++){
                            conv2d_8_bonus_output[i][j][k] += pointwiseconv2d_7_output[i][j][l] * conv2d_8_bonus_kernel[m][n][l][k];
                        }
                    }
                }
                conv2d_8_bonus_output[i][j][k] += conv2d_8_bonus_bias[k];
            }
        }
    }
    printf("Size of conv2d_8_bonus_output: %d x %d x %d \n", LEN(conv2d_8_bonus_output), LEN(conv2d_8_bonus_output[0]), LEN(conv2d_8_bonus_output[0][0]));
}

void depthwiseconv2d_8_bonus(){
    static int depthwiseconv2d_8_bonus_kernel[depthwiseconv2d_8_bonus_kernel_size][depthwiseconv2d_8_bonus_kernel_size][depthwiseconv2d_8_bonus_kernel_channels][depthwiseconv2d_8_bonus_kernel_num];
    static int depthwiseconv2d_8_bonus_bias[depthwiseconv2d_8_bonus_kernel_channels];
    for (int k = 0; k < depthwiseconv2d_8_bonus_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < depthwiseconv2d_8_bonus_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_8_bonus_kernel_size; j++){
                for(int l = 0; l < depthwiseconv2d_8_bonus_kernel_channels; l++){
                    depthwiseconv2d_8_bonus_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", depthwiseconv2d_8_bonus_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_8_bonus_kernel_channels; k++){
        depthwiseconv2d_8_bonus_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_8_bonus_bias[k]);
    }

    //Determine the padded conv2d_8_bonus_output size 
    static const int conv2d_8_bonus_padded_output_size = 16;
    static int conv2d_8_bonus_padded_output[16][16][depthwiseconv2d_8_bonus_kernel_channels];

    //Initialize the padded conv2d_8_bonus_output with 0
    for(int i = 0; i < conv2d_8_bonus_padded_output_size; i++){
        for(int j = 0; j < conv2d_8_bonus_padded_output_size; j++){
            for(int k = 0; k < depthwiseconv2d_8_bonus_kernel_channels; k++){
                conv2d_8_bonus_padded_output[i][j][k] = 0;
            }
        }
    }

    //Copy the conv2d_8_bonus_output to the padded conv2d_8_bonus_output
    for(int i = 1; i < depthwiseconv2d_8_bonus_output_size - 1; i++){
        for(int j = 1; j < depthwiseconv2d_8_bonus_output_size - 1; j++){
            for(int k = 0; k < depthwiseconv2d_8_bonus_kernel_channels; k++){
                conv2d_8_bonus_padded_output[i][j][k] = conv2d_8_bonus_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_8_bonus_padded_output: %d x %d x %d \n", LEN(conv2d_8_bonus_padded_output), LEN(conv2d_8_bonus_padded_output[0]), LEN(conv2d_8_bonus_padded_output[0][0]));
    
    //Perform depthwiseconv2d_8_bonus
    for(int k = 0; k < depthwiseconv2d_8_bonus_kernel_num; k++){
        for(int i = 0; i < depthwiseconv2d_8_bonus_output_size; i++){
            for(int j = 0; j < depthwiseconv2d_8_bonus_output_size; j++){
                depthwiseconv2d_8_bonus_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_8_bonus_kernel_channels; l++){
                    for(int m = 0; m < depthwiseconv2d_8_bonus_kernel_size; m++){
                        for(int n = 0; n < depthwiseconv2d_8_bonus_kernel_size; n++){
                            depthwiseconv2d_8_bonus_output[i][j][l] += conv2d_8_bonus_padded_output[i + m][j + n][l] * depthwiseconv2d_8_bonus_kernel[m][n][l][k];
                        }
                    }
                    depthwiseconv2d_8_bonus_output[i][j][k] += depthwiseconv2d_8_bonus_bias[l];
                }
            }
        }
    }
    printf("Size of depthwiseconv2d_8_bonus_output: %d x %d x %d \n", LEN(depthwiseconv2d_8_bonus_output), LEN(depthwiseconv2d_8_bonus_output[0]), LEN(depthwiseconv2d_8_bonus_output[0][0]));
}

void pointwiseconv2d_8_bonus(){
    static int pointwiseconv2d_8_bonus_kernel[pointwiseconv2d_8_bonus_kernel_size][pointwiseconv2d_8_bonus_kernel_size][pointwiseconv2d_8_bonus_kernel_channels][pointwiseconv2d_8_bonus_kernel_num];
    static int pointwiseconv2d_8_bonus_bias[pointwiseconv2d_8_bonus_kernel_num];
    for (int k = 0; k < pointwiseconv2d_8_bonus_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int l = 0; l < pointwiseconv2d_8_bonus_kernel_channels; l++){
            for (int i = 0; i < pointwiseconv2d_8_bonus_kernel_size; i++){
                for (int j = 0; j < pointwiseconv2d_8_bonus_kernel_size; j++){
                    pointwiseconv2d_8_bonus_kernel[i][j][l][k] = rand() % 5 - 2;
                }
            }
            // printf("%f ", pointwiseconv2d_8_bonus_kernel[l][k]);
        }
        // printf("\n");
    }
    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_8_bonus_kernel_num; k++){
        pointwiseconv2d_8_bonus_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_8_bonus_bias[k]);
    }

    //Perform pointwiseconv2d_8_bonus
    for(int k = 0; k < pointwiseconv2d_8_bonus_kernel_num; k++){
        for(int i = 0; i < pointwiseconv2d_8_bonus_output_size; i++){
            for(int j = 0; j < pointwiseconv2d_8_bonus_output_size; j++){
                pointwiseconv2d_8_bonus_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_8_bonus_kernel_channels; l++){
                    for(int m = 0; m < pointwiseconv2d_8_bonus_kernel_size; m++){
                        for(int n = 0; n < pointwiseconv2d_8_bonus_kernel_size; n++){
                            pointwiseconv2d_8_bonus_output[i][j][k] += depthwiseconv2d_8_bonus_output[i + m][j + n][l] * pointwiseconv2d_8_bonus_kernel[m][n][l][k];
                        }
                    }
                    pointwiseconv2d_8_bonus_output[i][j][k] += pointwiseconv2d_8_bonus_bias[k];
                }
            }
        }
    }
    printf("Size of pointwiseconv2d_8_bonus_output: %d x %d x %d \n", LEN(pointwiseconv2d_8_bonus_output), LEN(pointwiseconv2d_8_bonus_output[0]), LEN(pointwiseconv2d_8_bonus_output[0][0]));
}

void add_5(){
    for(int i = 0; i < add_5_output_size; i++){
        for(int j = 0; j < add_5_output_size; j++){
            for(int k = 0; k < add_5_output_channels; k++){
                add_5_output[i][j][k] = pointwiseconv2d_7_output[i][j][k] + pointwiseconv2d_8_bonus_output[i][j][k];
            }
        }
    }
    printf("Size of add_5_output: %d x %d x %d \n", LEN(add_5_output), LEN(add_5_output[0]), LEN(add_5_output[0][0]));
}

void conv2d_8(){
    static int conv2d_8_kernel[conv2d_8_kernel_size][conv2d_8_kernel_size][conv2d_8_kernel_channels][conv2d_8_kernel_num];
    static int conv2d_8_bias[conv2d_8_kernel_num];
    for (int k = 0; k < conv2d_8_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_8_kernel_size; i++){
            for(int j = 0; j < conv2d_8_kernel_size; j++){
                for(int l = 0; l < conv2d_8_kernel_channels; l++){
                    conv2d_8_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_8_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_8_kernel_num; k++){
        conv2d_8_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_8_bias[k]);
    }

    //Perform conv2d_8
    for(int k = 0; k < conv2d_8_kernel_num; k++){
        for(int i = 0; i < conv2d_8_output_size; i++){
            for(int j = 0; j < conv2d_8_output_size; j++){
                conv2d_8_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_8_kernel_channels; l++){
                    for(int m = 0; m < conv2d_8_kernel_size; m++){
                        for(int n = 0; n < conv2d_8_kernel_size; n++){
                            conv2d_8_output[i][j][k] += add_5_output[i + m][j + n][l] * conv2d_8_kernel[m][n][l][k];
                        }
                    }
                }
                conv2d_8_output[i][j][k] += conv2d_8_bias[k];
            }
        }
    }
    printf("Size of conv2d_8_output: %d x %d x %d \n", LEN(conv2d_8_output), LEN(conv2d_8_output[0]), LEN(conv2d_8_output[0][0]));
}

void depthwiseconv2d_8(){
    static int depthwiseconv2d_8_kernel[depthwiseconv2d_8_kernel_size][depthwiseconv2d_8_kernel_size][depthwiseconv2d_8_kernel_channels];
    static int depthwiseconv2d_8_bias[depthwiseconv2d_8_kernel_channels];
    for (int k = 0; k < depthwiseconv2d_8_kernel_channels; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < depthwiseconv2d_8_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_8_kernel_size; j++){
                depthwiseconv2d_8_kernel[i][j][k] = rand() % 5 - 2;
                // printf("%f ", depthwiseconv2d_8_kernel[i][j][k]);
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_8_kernel_channels; k++){
        depthwiseconv2d_8_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_8_bias[k]);
    }

    //Determine padded conv2d_8_output size
    static const int conv2d_8_output_padded_size = 16;
    static int conv2d_8_output_padded[16][16][depthwiseconv2d_8_kernel_channels];

    //Initialize padded conv2d_8_output with 0
    for(int i = 0; i < conv2d_8_output_padded_size; i++){
        for(int j = 0; j < conv2d_8_output_padded_size; j++){
            for(int k = 0; k < depthwiseconv2d_8_kernel_channels; k++){
                conv2d_8_output_padded[i][j][k] = 0;
            }
        }
    }
    //Copy conv2d_8_output to padded conv2d_8_output
    for(int i = 1; i < conv2d_8_output_size - 1; i++){
        for(int j = 1; j < conv2d_8_output_size - 1; j++){
            for(int k = 0; k < depthwiseconv2d_8_kernel_channels; k++){
                conv2d_8_output_padded[i][j][k] = conv2d_8_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_8_output_padded: %d x %d x %d \n", LEN(conv2d_8_output_padded), LEN(conv2d_8_output_padded[0]), LEN(conv2d_8_output_padded[0][0]));

    //Perform depthwiseconv2d_8
    for(int k = 0; k < depthwiseconv2d_8_kernel_channels; k++){
        for(int i = 0; i < conv2d_8_output_size; i++){
            for(int j = 0; j < conv2d_8_output_size; j++){
                depthwiseconv2d_8_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_8_kernel_size; l++){
                    for(int m = 0; m < depthwiseconv2d_8_kernel_size; m++){
                        depthwiseconv2d_8_output[i][j][k] += conv2d_8_output_padded[i + l][j + m][k] * depthwiseconv2d_8_kernel[l][m][k];
                    }
                }
                depthwiseconv2d_8_output[i][j][k] += depthwiseconv2d_8_bias[k];
            }
        }
    }
    printf("Size of depthwiseconv2d_8_output: %d x %d x %d \n", LEN(depthwiseconv2d_8_output), LEN(depthwiseconv2d_8_output[0]), LEN(depthwiseconv2d_8_output[0][0]));
}

void pointwiseconv2d_8(){
    static int pointwiseconv2d_8_kernel[pointwiseconv2d_8_kernel_size][pointwiseconv2d_8_kernel_size][pointwiseconv2d_8_kernel_channels][pointwiseconv2d_8_kernel_num];
    static int pointwiseconv2d_8_bias[pointwiseconv2d_8_kernel_num];
    for (int k = 0; k < pointwiseconv2d_8_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < pointwiseconv2d_8_kernel_size; i++){
            for(int j = 0; j < pointwiseconv2d_8_kernel_size; j++){
                for(int l = 0; l < pointwiseconv2d_8_kernel_channels; l++){
                    pointwiseconv2d_8_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_8_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_8_kernel_num; k++){
        pointwiseconv2d_8_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_8_bias[k]);
    }

    //Perform pointwiseconv2d_8
    for(int k = 0; k < pointwiseconv2d_8_kernel_num; k++){
        for(int i = 0; i < conv2d_8_output_size; i++){
            for(int j = 0; j < conv2d_8_output_size; j++){
                pointwiseconv2d_8_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_8_kernel_size; l++){
                    for(int m = 0; m < pointwiseconv2d_8_kernel_size; m++){
                        for(int n = 0; n < pointwiseconv2d_8_kernel_channels; n++){
                            pointwiseconv2d_8_output[i][j][k] += depthwiseconv2d_8_output[i + l][j + m][n] * pointwiseconv2d_8_kernel[l][m][n][k];
                        }
                    }
                }
                pointwiseconv2d_8_output[i][j][k] += pointwiseconv2d_8_bias[k];
            }
        }
    }

    printf("Size of pointwiseconv2d_8_output: %d x %d x %d \n", LEN(pointwiseconv2d_8_output), LEN(pointwiseconv2d_8_output[0]), LEN(pointwiseconv2d_8_output[0][0]));
}

void add_6(){
    //Perform add_6
    for(int i = 0; i < conv2d_8_output_size; i++){
        for(int j = 0; j < conv2d_8_output_size; j++){
            for(int k = 0; k < pointwiseconv2d_8_kernel_num; k++){
                add_6_output[i][j][k] = pointwiseconv2d_8_output[i][j][k] + add_5_output[i][j][k];
            }
        }
    }
    printf("Size of add_6_output: %d x %d x %d \n", LEN(add_6_output), LEN(add_6_output[0]), LEN(add_6_output[0][0]));
}

void conv2d_9(){
    static int conv2d_9_kernel[conv2d_9_kernel_size][conv2d_9_kernel_size][conv2d_9_kernel_channels][conv2d_9_kernel_num];
    static int conv2d_9_bias[conv2d_9_kernel_num];
    for (int k = 0; k < conv2d_9_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_9_kernel_size; i++){
            for(int j = 0; j < conv2d_9_kernel_size; j++){
                for(int l = 0; l < conv2d_9_kernel_channels; l++){
                    conv2d_9_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_9_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_9_kernel_num; k++){
        conv2d_9_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_9_bias[k]);
    }

    //Perform conv2d_9
    for(int k = 0; k < conv2d_9_kernel_num; k++){
        for(int i = 0; i < conv2d_9_output_size; i++){
            for(int j = 0; j < conv2d_9_output_size; j++){
                conv2d_9_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_9_kernel_size; l++){
                    for(int m = 0; m < conv2d_9_kernel_size; m++){
                        for(int n = 0; n < conv2d_9_kernel_channels; n++){
                            conv2d_9_output[i][j][k] += add_6_output[i + l][j + m][n] * conv2d_9_kernel[l][m][n][k];
                        }
                    }
                }
                conv2d_9_output[i][j][k] += conv2d_9_bias[k];
            }
        }
    }
    printf("Size of conv2d_9_output: %d x %d x %d \n", LEN(conv2d_9_output), LEN(conv2d_9_output[0]), LEN(conv2d_9_output[0][0]));
}

void depthwiseconv2d_9(){
    static int depthwiseconv2d_9_kernel[depthwiseconv2d_9_kernel_size][depthwiseconv2d_9_kernel_size][depthwiseconv2d_9_kernel_channels];
    static int depthwiseconv2d_9_bias[depthwiseconv2d_9_kernel_channels];
    for (int k = 0; k < depthwiseconv2d_9_kernel_channels; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < depthwiseconv2d_9_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_9_kernel_size; j++){
                depthwiseconv2d_9_kernel[i][j][k] = rand() % 5 - 2;
                // printf("%f ", depthwiseconv2d_9_kernel[i][j][k]);
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_9_kernel_channels; k++){
        depthwiseconv2d_9_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_9_bias[k]);
    }

    //Determine the padded conv2d_9_output size
    static const int conv2d_9_output_padded_size = 16;
    static int conv2d_9_output_padded[16][16][depthwiseconv2d_9_kernel_channels];

    //Initialize the padded conv2d_9_output with 0
    for(int i = 0; i < conv2d_9_output_padded_size; i++){
        for(int j = 0; j < conv2d_9_output_padded_size; j++){
            for(int k = 0; k < depthwiseconv2d_9_kernel_channels; k++){
                conv2d_9_output_padded[i][j][k] = 0;
            }
        }
    }

    //Copy conv2d_9_output to the padded conv2d_9_output
    for(int i = 1; i < conv2d_9_output_size - 1; i++){
        for(int j = 1; j < conv2d_9_output_size - 1; j++){
            for(int k = 0; k < depthwiseconv2d_9_kernel_channels; k++){
                conv2d_9_output_padded[i][j][k] = conv2d_9_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_9_output_padded: %d x %d x %d \n", LEN(conv2d_9_output_padded), LEN(conv2d_9_output_padded[0]), LEN(conv2d_9_output_padded[0][0]));

    //Perform depthwiseconv2d_9
    for(int k = 0; k < depthwiseconv2d_9_kernel_channels; k++){
        for(int i = 0; i < depthwiseconv2d_9_output_size; i++){
            for(int j = 0; j < depthwiseconv2d_9_output_size; j++){
                depthwiseconv2d_9_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_9_kernel_size; l++){
                    for(int m = 0; m < depthwiseconv2d_9_kernel_size; m++){
                        depthwiseconv2d_9_output[i][j][k] += conv2d_9_output_padded[i + l][j + m][k] * depthwiseconv2d_9_kernel[l][m][k];
                    }
                }
                depthwiseconv2d_9_output[i][j][k] += depthwiseconv2d_9_bias[k];
            }
        }
    }

    printf("Size of depthwiseconv2d_9_output: %d x %d x %d \n", LEN(depthwiseconv2d_9_output), LEN(depthwiseconv2d_9_output[0]), LEN(depthwiseconv2d_9_output[0][0]));
}

void pointwiseconv2d_9(){
    static int pointwiseconv2d_9_kernel[pointwiseconv2d_9_kernel_size][pointwiseconv2d_9_kernel_size][pointwiseconv2d_9_kernel_channels][pointwiseconv2d_9_kernel_num];
    static int pointwiseconv2d_9_bias[pointwiseconv2d_9_kernel_num];
    for (int k = 0; k < pointwiseconv2d_9_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < pointwiseconv2d_9_kernel_size; i++){
            for(int j = 0; j < pointwiseconv2d_9_kernel_size; j++){
                for(int l = 0; l < pointwiseconv2d_9_kernel_channels; l++){
                    pointwiseconv2d_9_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_9_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_9_kernel_num; k++){
        pointwiseconv2d_9_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_9_bias[k]);
    }

    //Perform pointwiseconv2d_9
    for(int k = 0; k < pointwiseconv2d_9_kernel_num; k++){
        for(int i = 0; i < pointwiseconv2d_9_output_size; i++){
            for(int j = 0; j < pointwiseconv2d_9_output_size; j++){
                pointwiseconv2d_9_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_9_kernel_channels; l++){
                    pointwiseconv2d_9_output[i][j][k] += depthwiseconv2d_9_output[i][j][l] * pointwiseconv2d_9_kernel[0][0][l][k];
                }
                pointwiseconv2d_9_output[i][j][k] += pointwiseconv2d_9_bias[k];
            }
        }
    }

    printf("Size of pointwiseconv2d_9_output: %d x %d x %d \n", LEN(pointwiseconv2d_9_output), LEN(pointwiseconv2d_9_output[0]), LEN(pointwiseconv2d_9_output[0][0]));
}

void add_7(){
    //Perform add_7
    for(int i = 0; i < add_7_output_size; i++){
        for(int j = 0; j < add_7_output_size; j++){
            for(int k = 0; k < add_7_output_channels; k++){
                add_7_output[i][j][k] = pointwiseconv2d_9_output[i][j][k] + add_6_output[i][j][k];
            }
        }
    }

    printf("Size of add_7_output: %d x %d x %d \n", LEN(add_7_output), LEN(add_7_output[0]), LEN(add_7_output[0][0]));
}

void conv2d_10(){
    static int conv2d_10_kernel[conv2d_10_kernel_size][conv2d_10_kernel_size][conv2d_10_kernel_channels][conv2d_10_kernel_num];
    static int conv2d_10_bias[conv2d_10_kernel_num];
    for (int k = 0; k < conv2d_10_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_10_kernel_size; i++){
            for(int j = 0; j < conv2d_10_kernel_size; j++){
                for(int l = 0; l < conv2d_10_kernel_channels; l++){
                    conv2d_10_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_10_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_10_kernel_num; k++){
        conv2d_10_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_10_bias[k]);
    }

    //Perform conv2d_10
    for(int k = 0; k < conv2d_10_kernel_num; k++){
        for(int i = 0; i < conv2d_10_output_size; i++){
            for(int j = 0; j < conv2d_10_output_size; j++){
                conv2d_10_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_10_kernel_channels; l++){
                    for(int m = 0; m < conv2d_10_kernel_size; m++){
                        for(int n = 0; n < conv2d_10_kernel_size; n++){
                            conv2d_10_output[i][j][k] += add_7_output[i + m][j + n][l] * conv2d_10_kernel[m][n][l][k];
                        }
                    }
                }
                conv2d_10_output[i][j][k] += conv2d_10_bias[k];
            }
        }
    }

    printf("Size of conv2d_10_output: %d x %d x %d \n", LEN(conv2d_10_output), LEN(conv2d_10_output[0]), LEN(conv2d_10_output[0][0]));
}

void depthwiseconv2d_10(){
    static int depthwiseconv2d_10_kernel[depthwiseconv2d_10_kernel_size][depthwiseconv2d_10_kernel_size][depthwiseconv2d_10_kernel_channels][depthwiseconv2d_10_kernel_num];
    static int depthwiseconv2d_10_bias[depthwiseconv2d_10_kernel_channels];
    for (int k = 0; k < depthwiseconv2d_10_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < depthwiseconv2d_10_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_10_kernel_size; j++){
                for(int l = 0; l < depthwiseconv2d_10_kernel_channels; l++){
                    depthwiseconv2d_10_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", depthwiseconv2d_10_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_10_kernel_channels; k++){
        depthwiseconv2d_10_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_10_bias[k]);
    }

    //Determine the padded conv2d_10_output size
    static const int conv2d_10_output_padded_size = 16;
    static int conv2d_10_output_padded[16][16][depthwiseconv2d_10_kernel_channels];

    //Initialize the padded conv2d_10_output with 0
    for(int i = 0; i < conv2d_10_output_padded_size; i++){
        for(int j = 0; j < conv2d_10_output_padded_size; j++){
            for(int k = 0; k < depthwiseconv2d_10_kernel_channels; k++){
                conv2d_10_output_padded[i][j][k] = 0;
            }
        }
    }

    //Copy the conv2d_10_output to the padded conv2d_10_output
    for(int i = 1; i < conv2d_10_output_size - 1; i++){
        for(int j = 1; j < conv2d_10_output_size - 1; j++){
            for(int k = 0; k < depthwiseconv2d_10_kernel_channels; k++){
                conv2d_10_output_padded[i][j][k] = conv2d_10_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_10_output_padded: %d x %d x %d \n", LEN(conv2d_10_output_padded), LEN(conv2d_10_output_padded[0]), LEN(conv2d_10_output_padded[0][0]));

    //Perform depthwiseconv2d_10
    for(int k = 0; k < depthwiseconv2d_10_kernel_channels; k++){
        for(int i = 0; i < depthwiseconv2d_10_output_size; i++){
            for(int j = 0; j < depthwiseconv2d_10_output_size; j++){
                depthwiseconv2d_10_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_10_kernel_size; l++){
                    for(int m = 0; m < depthwiseconv2d_10_kernel_size; m++){
                        depthwiseconv2d_10_output[i][j][k] += conv2d_10_output_padded[i + l][j + m][k] * depthwiseconv2d_10_kernel[l][m][k][k];
                    }
                }
                depthwiseconv2d_10_output[i][j][k] += depthwiseconv2d_10_bias[k];
            }
        }
    }
    printf("Size of depthwiseconv2d_10_output: %d x %d x %d \n", LEN(depthwiseconv2d_10_output), LEN(depthwiseconv2d_10_output[0]), LEN(depthwiseconv2d_10_output[0][0]));
}

void pointwiseconv2d_10(){
    static int pointwiseconv2d_10_kernel[pointwiseconv2d_10_kernel_size][pointwiseconv2d_10_kernel_size][pointwiseconv2d_10_kernel_channels][pointwiseconv2d_10_kernel_num];
    static int pointwiseconv2d_10_bias[pointwiseconv2d_10_kernel_num];
    for (int k = 0; k < pointwiseconv2d_10_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < pointwiseconv2d_10_kernel_size; i++){
            for(int j = 0; j < pointwiseconv2d_10_kernel_size; j++){
                for(int l = 0; l < pointwiseconv2d_10_kernel_channels; l++){
                    pointwiseconv2d_10_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_10_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_10_kernel_num; k++){
        pointwiseconv2d_10_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_10_bias[k]);
    }

    //Perform pointwiseconv2d_10
    for(int k = 0; k < pointwiseconv2d_10_kernel_num; k++){
        for(int i = 0; i < pointwiseconv2d_10_output_size; i++){
            for(int j = 0; j < pointwiseconv2d_10_output_size; j++){
                pointwiseconv2d_10_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_10_kernel_channels; l++){
                    pointwiseconv2d_10_output[i][j][k] += depthwiseconv2d_10_output[i][j][l] * pointwiseconv2d_10_kernel[0][0][l][k];
                }
                pointwiseconv2d_10_output[i][j][k] += pointwiseconv2d_10_bias[k];
            }
        }
    }

    printf("Size of pointwiseconv2d_10_output: %d x %d x %d \n", LEN(pointwiseconv2d_10_output), LEN(pointwiseconv2d_10_output[0]), LEN(pointwiseconv2d_10_output[0][0]));
}

void conv2d_11(){
    static int conv2d_11_kernel[conv2d_11_kernel_size][conv2d_11_kernel_size][conv2d_11_kernel_channels][conv2d_11_kernel_num];
    static int conv2d_11_bias[conv2d_11_kernel_num];
    for (int k = 0; k < conv2d_11_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_11_kernel_size; i++){
            for(int j = 0; j < conv2d_11_kernel_size; j++){
                for(int l = 0; l < conv2d_11_kernel_channels; l++){
                    conv2d_11_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_11_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_11_kernel_num; k++){
        conv2d_11_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_11_bias[k]);
    }
    //Perform conv2d_11
    for(int k = 0; k < conv2d_11_kernel_num; k++){
        for(int i = 0; i < conv2d_11_output_size; i++){
            for(int j = 0; j < conv2d_11_output_size; j++){
                conv2d_11_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_11_kernel_size; l++){
                    for(int m = 0; m < conv2d_11_kernel_size; m++){
                        for(int n = 0; n < conv2d_11_kernel_channels; n++){
                            conv2d_11_output[i][j][k] += pointwiseconv2d_10_output[i + l][j + m][n] * conv2d_11_kernel[l][m][n][k];
                        }
                    }
                }
                conv2d_11_output[i][j][k] += conv2d_11_bias[k];
            }
        }
    }
    printf("Size of conv2d_11_output: %d x %d x %d \n", LEN(conv2d_11_output), LEN(conv2d_11_output[0]), LEN(conv2d_11_output[0][0]));
}

void depthwiseconv2d_11(){
    static int depthwiseconv2d_11_kernel[depthwiseconv2d_11_kernel_size][depthwiseconv2d_11_kernel_size][depthwiseconv2d_11_kernel_channels];
    static int depthwiseconv2d_11_bias[depthwiseconv2d_11_kernel_channels];
    for (int k = 0; k < depthwiseconv2d_11_kernel_channels; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < depthwiseconv2d_11_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_11_kernel_size; j++){
                depthwiseconv2d_11_kernel[i][j][k] = rand() % 5 - 2;
                // printf("%f ", depthwiseconv2d_11_kernel[i][j][k]);
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_11_kernel_channels; k++){
        depthwiseconv2d_11_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_11_bias[k]);
    }

    //Determine the padded conv2d_11_output size
    static const int conv2d_11_output_padded_size = 16;
    static int conv2d_11_output_padded[16][16][depthwiseconv2d_11_kernel_channels];

    //Copy conv2d_11_output to conv2d_11_output_padded
    for(int k = 0; k < depthwiseconv2d_11_kernel_channels; k++){
        for(int i = 1; i < conv2d_11_output_size - 1; i++){
            for(int j = 1; j < conv2d_11_output_size - 1; j++){
                conv2d_11_output_padded[i][j][k] = conv2d_11_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_11_output_padded: %d x %d x %d \n", LEN(conv2d_11_output_padded), LEN(conv2d_11_output_padded[0]), LEN(conv2d_11_output_padded[0][0]));

    //Perform depthwiseconv2d_11
    for(int k = 0; k < depthwiseconv2d_11_kernel_channels; k++){
        for(int i = 0; i < depthwiseconv2d_11_output_size; i++){
            for(int j = 0; j < depthwiseconv2d_11_output_size; j++){
                depthwiseconv2d_11_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_11_kernel_size; l++){
                    for(int m = 0; m < depthwiseconv2d_11_kernel_size; m++){
                        depthwiseconv2d_11_output[i][j][k] += conv2d_11_output_padded[i + l][j + m][k] * depthwiseconv2d_11_kernel[l][m][k];
                    }
                }
                depthwiseconv2d_11_output[i][j][k] += depthwiseconv2d_11_bias[k];
            }
        }
    }

    printf("Size of depthwiseconv2d_11_output: %d x %d x %d \n", LEN(depthwiseconv2d_11_output), LEN(depthwiseconv2d_11_output[0]), LEN(depthwiseconv2d_11_output[0][0]));
}

void pointwiseconv2d_11(){
    static int pointwiseconv2d_11_kernel[pointwiseconv2d_11_kernel_size][pointwiseconv2d_11_kernel_size][pointwiseconv2d_11_kernel_channels][pointwiseconv2d_11_kernel_num];
    static int pointwiseconv2d_11_bias[pointwiseconv2d_11_kernel_num];
    for (int k = 0; k < pointwiseconv2d_11_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < pointwiseconv2d_11_kernel_size; i++){
            for(int j = 0; j < pointwiseconv2d_11_kernel_size; j++){
                for(int l = 0; l < pointwiseconv2d_11_kernel_channels; l++){
                    pointwiseconv2d_11_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_11_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_11_kernel_num; k++){
        pointwiseconv2d_11_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_11_bias[k]);
    }
    //Perform pointwiseconv2d_11
    for(int k = 0; k < pointwiseconv2d_11_kernel_num; k++){
        for(int i = 0; i < pointwiseconv2d_11_output_size; i++){
            for(int j = 0; j < pointwiseconv2d_11_output_size; j++){
                pointwiseconv2d_11_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_11_kernel_size; l++){
                    for(int m = 0; m < pointwiseconv2d_11_kernel_size; m++){
                        for(int n = 0; n < pointwiseconv2d_11_kernel_channels; n++){
                            pointwiseconv2d_11_output[i][j][k] += depthwiseconv2d_11_output[i + l][j + m][n] * pointwiseconv2d_11_kernel[l][m][n][k];
                        }
                    }
                }
                pointwiseconv2d_11_output[i][j][k] += pointwiseconv2d_11_bias[k];
            }
        }
    }
    printf("Size of pointwiseconv2d_11_output: %d x %d x %d \n", LEN(pointwiseconv2d_11_output), LEN(pointwiseconv2d_11_output[0]), LEN(pointwiseconv2d_11_output[0][0]));
}

void add_8(){
    //Perform add_8
    for(int k = 0; k < add_8_output_channels; k++){
        for(int i = 0; i < add_8_output_size; i++){
            for(int j = 0; j < add_8_output_size; j++){
                add_8_output[i][j][k] = pointwiseconv2d_11_output[i][j][k] + pointwiseconv2d_10_output[i][j][k];
            }
        }
    }
    printf("Size of add_8_output: %d x %d x %d \n", LEN(add_8_output), LEN(add_8_output[0]), LEN(add_8_output[0][0]));
}

void conv2d_12(){
    static int conv2d_12_kernel[conv2d_12_kernel_size][conv2d_12_kernel_size][conv2d_12_kernel_channels][conv2d_12_kernel_num];
    static int conv2d_12_bias[conv2d_12_kernel_num];
    for (int k = 0; k < conv2d_12_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_12_kernel_size; i++){
            for(int j = 0; j < conv2d_12_kernel_size; j++){
                for(int l = 0; l < conv2d_12_kernel_channels; l++){
                    conv2d_12_kernel[i][j][l][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_12_kernel[i][j][l][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_12_kernel_num; k++){
        conv2d_12_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_12_bias[k]);
    }
    //Perform conv2d_12
    for(int k = 0; k < conv2d_12_kernel_num; k++){
        for(int i = 0; i < conv2d_12_output_size; i++){
            for(int j = 0; j < conv2d_12_output_size; j++){
                conv2d_12_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_12_kernel_size; l++){
                    for(int m = 0; m < conv2d_12_kernel_size; m++){
                        for(int n = 0; n < conv2d_12_kernel_channels; n++){
                            conv2d_12_output[i][j][k] += add_8_output[i + l][j + m][n] * conv2d_12_kernel[l][m][n][k];
                        }
                    }
                }
                conv2d_12_output[i][j][k] += conv2d_12_bias[k];
            }
        }
    }
    printf("Size of conv2d_12_output: %d x %d x %d \n", LEN(conv2d_12_output), LEN(conv2d_12_output[0]), LEN(conv2d_12_output[0][0]));
}

void depthwiseconv2d_12(){
    static int depthwiseconv2d_12_kernel[depthwiseconv2d_12_kernel_size][depthwiseconv2d_12_kernel_size][depthwiseconv2d_12_kernel_channels];
    static int depthwiseconv2d_12_bias[depthwiseconv2d_12_kernel_channels];
    for (int k = 0; k < depthwiseconv2d_12_kernel_channels; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < depthwiseconv2d_12_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_12_kernel_size; j++){
                depthwiseconv2d_12_kernel[i][j][k] = rand() % 5 - 2;
                // printf("%f ", depthwiseconv2d_12_kernel[i][j][k]);
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_12_kernel_channels; k++){
        depthwiseconv2d_12_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_12_bias[k]);
    }

    //Determine padded conv2d_12_output size
    static const int conv2d_12_output_padded_size = 15;
    static int conv2d_12_output_padded[15][15][conv2d_12_kernel_num];

    //Initialize conv2d_12_output_padded with 0
    for(int k = 0; k < conv2d_12_kernel_num; k++){
        for(int i = 0; i < conv2d_12_output_padded_size; i++){
            for(int j = 0; j < conv2d_12_output_padded_size; j++){
                conv2d_12_output_padded[i][j][k] = 0;
            }
        }
    }
    //Copy conv2d_12_output to conv2d_12_output_padded
    for(int k = 0; k < conv2d_12_kernel_num; k++){
        for(int i = 0; i < conv2d_12_output_size; i++){
            for(int j = 0; j < conv2d_12_output_size; j++){
                conv2d_12_output_padded[i][j][k] = conv2d_12_output[i][j][k];
            }
        }
    }
    printf("Size of conv2d_12_output_padded: %d x %d x %d \n", LEN(conv2d_12_output_padded), LEN(conv2d_12_output_padded[0]), LEN(conv2d_12_output_padded[0][0]));
    //Perform depthwiseconv2d_12
    for(int k = 0; k < depthwiseconv2d_12_kernel_channels; k++){
        for(int i = 0; i < depthwiseconv2d_12_output_size; i++){
            for(int j = 0; j < depthwiseconv2d_12_output_size; j++){
                depthwiseconv2d_12_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_12_kernel_size; l++){
                    for(int m = 0; m < depthwiseconv2d_12_kernel_size; m++){
                        depthwiseconv2d_12_output[i][j][k] += conv2d_12_output_padded[i + l][j + m][k] * depthwiseconv2d_12_kernel[l][m][k];
                    }
                }
                depthwiseconv2d_12_output[i][j][k] += depthwiseconv2d_12_bias[k];
            }
        }
    }
    printf("Size of depthwiseconv2d_12_output: %d x %d x %d \n", LEN(depthwiseconv2d_12_output), LEN(depthwiseconv2d_12_output[0]), LEN(depthwiseconv2d_12_output[0][0]));
}

void pointwiseconv2d_12(){
    static int pointwiseconv2d_12_kernel[pointwiseconv2d_12_kernel_size][pointwiseconv2d_12_kernel_size][pointwiseconv2d_12_kernel_channels][pointwiseconv2d_12_kernel_num];
    static int pointwiseconv2d_12_bias[pointwiseconv2d_12_kernel_num];
    for (int k = 0; k < pointwiseconv2d_12_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < pointwiseconv2d_12_kernel_size; i++){
            for(int j = 0; j < pointwiseconv2d_12_kernel_size; j++){
                for(int n = 0; n < pointwiseconv2d_12_kernel_channels; n++){
                    pointwiseconv2d_12_kernel[i][j][n][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_12_kernel[i][j][n][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_12_kernel_num; k++){
        pointwiseconv2d_12_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_12_bias[k]);
    }

    //Perform pointwiseconv2d_12
    for(int k = 0; k < pointwiseconv2d_12_kernel_num; k++){
        for(int i = 0; i < pointwiseconv2d_12_output_size; i++){
            for(int j = 0; j < pointwiseconv2d_12_output_size; j++){
                pointwiseconv2d_12_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_12_kernel_size; l++){
                    for(int m = 0; m < pointwiseconv2d_12_kernel_size; m++){
                        for(int n = 0; n < pointwiseconv2d_12_kernel_channels; n++){
                            pointwiseconv2d_12_output[i][j][k] += depthwiseconv2d_12_output[i + l][j + m][n] * pointwiseconv2d_12_kernel[l][m][n][k];
                        }
                    }
                }
                pointwiseconv2d_12_output[i][j][k] += pointwiseconv2d_12_bias[k];
            }
        }
    }
    printf("Size of pointwiseconv2d_12_output: %d x %d x %d \n", LEN(pointwiseconv2d_12_output), LEN(pointwiseconv2d_12_output[0]), LEN(pointwiseconv2d_12_output[0][0]));
}

void conv2d_13(){
    static int conv2d_13_kernel[conv2d_13_kernel_size][conv2d_13_kernel_size][conv2d_13_kernel_channels][conv2d_13_kernel_num];
    static int conv2d_13_bias[conv2d_13_kernel_num];
    for (int k = 0; k < conv2d_13_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_13_kernel_size; i++){
            for(int j = 0; j < conv2d_13_kernel_size; j++){
                for(int n = 0; n < conv2d_13_kernel_channels; n++){
                    conv2d_13_kernel[i][j][n][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_13_kernel[i][j][n][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_13_kernel_num; k++){
        conv2d_13_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_13_bias[k]);
    }

    //Perform conv2d_13
    for(int k = 0; k < conv2d_13_kernel_num; k++){
        for(int i = 0; i < conv2d_13_output_size; i++){
            for(int j = 0; j < conv2d_13_output_size; j++){
                conv2d_13_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_13_kernel_size; l++){
                    for(int m = 0; m < conv2d_13_kernel_size; m++){
                        for(int n = 0; n < conv2d_13_kernel_channels; n++){
                            conv2d_13_output[i][j][k] += pointwiseconv2d_12_output[i + l][j + m][n] * conv2d_13_kernel[l][m][n][k];
                        }
                    }
                }
                conv2d_13_output[i][j][k] += conv2d_13_bias[k];
            }
        }
    }
    printf("Size of conv2d_13_output: %d x %d x %d \n", LEN(conv2d_13_output), LEN(conv2d_13_output[0]), LEN(conv2d_13_output[0][0]));
}

void depthwiseconv2d_13(){
    static int depthwiseconv2d_13_kernel[depthwiseconv2d_13_kernel_size][depthwiseconv2d_13_kernel_size][depthwiseconv2d_13_kernel_channels];
    static int depthwiseconv2d_13_bias[depthwiseconv2d_13_kernel_channels];
    for (int k = 0; k < depthwiseconv2d_13_kernel_channels; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < depthwiseconv2d_13_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_13_kernel_size; j++){
                depthwiseconv2d_13_kernel[i][j][k] = rand() % 5 - 2;
                // printf("%f ", depthwiseconv2d_13_kernel[i][j][k]);
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_13_kernel_channels; k++){
        depthwiseconv2d_13_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_13_bias[k]);
    }

    //Determine the padded conv2d_13_output size 
    static const int conv2d_13_output_padded_size = 9;
    static int conv2d_13_output_padded[9][9][conv2d_13_kernel_num];

    //Initialize the padded conv2d_13_output with 0
    for(int k = 0; k < conv2d_13_kernel_num; k++){
        for(int i = 0; i < conv2d_13_output_padded_size; i++){
            for(int j = 0; j < conv2d_13_output_padded_size; j++){
                conv2d_13_output_padded[i][j][k] = 0;
            }
        }
    }
    //Copy the conv2d_13_output to the padded conv2d_13_output
    for(int k = 0; k < conv2d_13_kernel_num; k++){
        for(int i = 1; i < conv2d_13_output_size - 1; i++){
            for(int j = 1; j < conv2d_13_output_size - 1; j++){
                conv2d_13_output_padded[i][j][k] = conv2d_13_output[i][j][k];
            }
        }
    }
    printf("Size of conv2d_13_output_padded: %d x %d x %d \n", LEN(conv2d_13_output_padded), LEN(conv2d_13_output_padded[0]), LEN(conv2d_13_output_padded[0][0]));

    //Perform depthwiseconv2d_13
    for(int k = 0; k < depthwiseconv2d_13_kernel_channels; k++){
        for(int i = 0; i < depthwiseconv2d_13_output_size; i++){
            for(int j = 0; j < depthwiseconv2d_13_output_size; j++){
                depthwiseconv2d_13_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_13_kernel_size; l++){
                    for(int m = 0; m < depthwiseconv2d_13_kernel_size; m++){
                        depthwiseconv2d_13_output[i][j][k] += conv2d_13_output[i + l][j + m][k] * depthwiseconv2d_13_kernel[l][m][k];
                    }
                }
                depthwiseconv2d_13_output[i][j][k] += depthwiseconv2d_13_bias[k];
            }
        }
    }
    printf("Size of depthwiseconv2d_13_output: %d x %d x %d \n", LEN(depthwiseconv2d_13_output), LEN(depthwiseconv2d_13_output[0]), LEN(depthwiseconv2d_13_output[0][0]));
}

void pointwiseconv2d_13(){
    static int pointwiseconv2d_13_kernel[pointwiseconv2d_13_kernel_size][pointwiseconv2d_13_kernel_size][pointwiseconv2d_13_kernel_channels][pointwiseconv2d_13_kernel_num];
    static int pointwiseconv2d_13_bias[pointwiseconv2d_13_kernel_num];
    for (int k = 0; k < pointwiseconv2d_13_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < pointwiseconv2d_13_kernel_size; i++){
            for(int j = 0; j < pointwiseconv2d_13_kernel_size; j++){
                for(int n = 0; n < pointwiseconv2d_13_kernel_channels; n++){
                    pointwiseconv2d_13_kernel[i][j][n][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_13_kernel[i][j][n][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_13_kernel_num; k++){
        pointwiseconv2d_13_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_13_bias[k]);
    }

    //Perform pointwiseconv2d_13
    for(int k = 0; k < pointwiseconv2d_13_kernel_num; k++){
        for(int i = 0; i < pointwiseconv2d_13_output_size; i++){
            for(int j = 0; j < pointwiseconv2d_13_output_size; j++){
                pointwiseconv2d_13_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_13_kernel_size; l++){
                    for(int m = 0; m < pointwiseconv2d_13_kernel_size; m++){
                        for(int n = 0; n < pointwiseconv2d_13_kernel_channels; n++){
                            pointwiseconv2d_13_output[i][j][k] += depthwiseconv2d_13_output[i + l][j + m][n] * pointwiseconv2d_13_kernel[l][m][n][k];
                        }
                    }
                }
            }
        }
    }
    printf("Size of pointwiseconv2d_13_output: %d x %d x %d \n", LEN(pointwiseconv2d_13_output), LEN(pointwiseconv2d_13_output[0]), LEN(pointwiseconv2d_13_output[0][0]));
}

void add_9(){
    //Perform add_9
    for(int k = 0; k < add_9_output_channels; k++){
        for(int i = 0; i < add_9_output_size; i++){
            for(int j = 0; j < add_9_output_size; j++){
                add_9_output[i][j][k] = pointwiseconv2d_13_output[i][j][k] + pointwiseconv2d_12_output[i][j][k];
            }
        }
    }
    printf("Size of add_9_output: %d x %d x %d \n", LEN(add_9_output), LEN(add_9_output[0]), LEN(add_9_output[0][0]));
}

void conv2d_14(){
    static int conv2d_14_kernel[conv2d_14_kernel_size][conv2d_14_kernel_size][conv2d_14_kernel_channels][conv2d_14_kernel_num];
    static int conv2d_14_bias[conv2d_14_kernel_num];
    for (int k = 0; k < conv2d_14_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_14_kernel_size; i++){
            for(int j = 0; j < conv2d_14_kernel_size; j++){
                for(int n = 0; n < conv2d_14_kernel_channels; n++){
                    conv2d_14_kernel[i][j][n][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_14_kernel[i][j][n][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_14_kernel_num; k++){
        conv2d_14_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_14_bias[k]);
    }

    //Perform conv2d_14
    for(int k = 0; k < conv2d_14_kernel_num; k++){
        for(int i = 0; i < conv2d_14_output_size; i++){
            for(int j = 0; j < conv2d_14_output_size; j++){
                conv2d_14_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_14_kernel_size; l++){
                    for(int m = 0; m < conv2d_14_kernel_size; m++){
                        for(int n = 0; n < conv2d_14_kernel_channels; n++){
                            conv2d_14_output[i][j][k] += add_9_output[i + l][j + m][n] * conv2d_14_kernel[l][m][n][k];
                        }
                    }
                }
                conv2d_14_output[i][j][k] += conv2d_14_bias[k];
            }
        }
    }
    printf("Size of conv2d_14_output: %d x %d x %d \n", LEN(conv2d_14_output), LEN(conv2d_14_output[0]), LEN(conv2d_14_output[0][0]));
}

void depthwiseconv2d_14(){
    static int depthwiseconv2d_14_kernel[depthwiseconv2d_14_kernel_size][depthwiseconv2d_14_kernel_size][depthwiseconv2d_14_kernel_channels];
    static int depthwiseconv2d_14_bias[depthwiseconv2d_14_kernel_channels];
    for (int k = 0; k < depthwiseconv2d_14_kernel_channels; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < depthwiseconv2d_14_kernel_size; i++){
            for(int j = 0; j < depthwiseconv2d_14_kernel_size; j++){
                depthwiseconv2d_14_kernel[i][j][k] = rand() % 5 - 2;
                // printf("%f ", depthwiseconv2d_14_kernel[i][j][k]);
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < depthwiseconv2d_14_kernel_channels; k++){
        depthwiseconv2d_14_bias[k] = rand() % 5 - 2;
        // printf("%f ", depthwiseconv2d_14_bias[k]);
    }

    //Determine the padded conv2d_14_output size
    static const int conv2d_14_output_padded_size = 9;
    static int conv2d_14_output_padded[9][9][conv2d_14_kernel_num];

    //Initialize the padded conv2d_14_output with 0
    for(int k = 0; k < conv2d_14_kernel_num; k++){
        for(int i = 0; i < conv2d_14_output_padded_size; i++){
            for(int j = 0; j < conv2d_14_output_padded_size; j++){
                conv2d_14_output_padded[i][j][k] = 0;
            }
        }
    }

    //Copy the conv2d_14_output to the padded conv2d_14_output
    for(int k = 0; k < conv2d_14_kernel_num; k++){
        for(int i = 1; i < conv2d_14_output_size - 1; i++){
            for(int j = 1; j < conv2d_14_output_size - 1; j++){
                conv2d_14_output_padded[i][j][k] = conv2d_14_output[i][j][k];
            }
        }
    }
    printf("Size of conv2d_14_output_padded: %d x %d x %d \n", LEN(conv2d_14_output_padded), LEN(conv2d_14_output_padded[0]), LEN(conv2d_14_output_padded[0][0]));

    //Perform depthwiseconv2d_14
    for(int k = 0; k < depthwiseconv2d_14_kernel_channels; k++){
        for(int i = 0; i < depthwiseconv2d_14_output_size; i++){
            for(int j = 0; j < depthwiseconv2d_14_output_size; j++){
                depthwiseconv2d_14_output[i][j][k] = 0;
                for(int l = 0; l < depthwiseconv2d_14_kernel_size; l++){
                    for(int m = 0; m < depthwiseconv2d_14_kernel_size; m++){
                        depthwiseconv2d_14_output[i][j][k] += conv2d_14_output[i + l][j + m][k] * depthwiseconv2d_14_kernel[l][m][k];
                    }
                }
                depthwiseconv2d_14_output[i][j][k] += depthwiseconv2d_14_bias[k];
            }
        }
    }
    printf("Size of depthwiseconv2d_14_output: %d x %d x %d \n", LEN(depthwiseconv2d_14_output), LEN(depthwiseconv2d_14_output[0]), LEN(depthwiseconv2d_14_output[0][0]));
}

void pointwiseconv2d_14(){
    static int pointwiseconv2d_14_kernel[pointwiseconv2d_14_kernel_size][pointwiseconv2d_14_kernel_size][pointwiseconv2d_14_kernel_channels][pointwiseconv2d_14_kernel_num];
    static int pointwiseconv2d_14_bias[pointwiseconv2d_14_kernel_num];
    for (int k = 0; k < pointwiseconv2d_14_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < pointwiseconv2d_14_kernel_size; i++){
            for(int j = 0; j < pointwiseconv2d_14_kernel_size; j++){
                for(int n = 0; n < pointwiseconv2d_14_kernel_channels; n++){
                    pointwiseconv2d_14_kernel[i][j][n][k] = rand() % 5 - 2;
                    // printf("%f ", pointwiseconv2d_14_kernel[i][j][n][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < pointwiseconv2d_14_kernel_num; k++){
        pointwiseconv2d_14_bias[k] = rand() % 5 - 2;
        // printf("%f ", pointwiseconv2d_14_bias[k]);
    }

    //Perform pointwiseconv2d_14
    for(int k = 0; k < pointwiseconv2d_14_kernel_num; k++){
        for(int i = 0; i < pointwiseconv2d_14_output_size; i++){
            for(int j = 0; j < pointwiseconv2d_14_output_size; j++){
                pointwiseconv2d_14_output[i][j][k] = 0;
                for(int l = 0; l < pointwiseconv2d_14_kernel_size; l++){
                    for(int m = 0; m < pointwiseconv2d_14_kernel_size; m++){
                        for(int n = 0; n < pointwiseconv2d_14_kernel_channels; n++){
                            pointwiseconv2d_14_output[i][j][k] += depthwiseconv2d_14_output[i + l][j + m][n] * pointwiseconv2d_14_kernel[l][m][n][k];
                        }
                    }
                }
                pointwiseconv2d_14_output[i][j][k] += pointwiseconv2d_14_bias[k];
            }
        }
    }
    printf("Size of pointwiseconv2d_14_output: %d x %d x %d \n", LEN(pointwiseconv2d_14_output), LEN(pointwiseconv2d_14_output[0]), LEN(pointwiseconv2d_14_output[0][0]));
}

void add_10(){
    //Perform add_10
    for(int k = 0; k < add_10_output_channels; k++){
        for(int i = 0; i < add_10_output_size; i++){
            for(int j = 0; j < add_10_output_size; j++){
                add_10_output[i][j][k] = pointwiseconv2d_14_output[i][j][k] + add_9_output[i][j][k];
            }
        }
    }
    printf("Size of add_10_output: %d x %d x %d \n", LEN(add_10_output), LEN(add_10_output[0]), LEN(add_10_output[0][0]));
}

void conv2d_15(){
    static int conv2d_15_kernel[conv2d_15_kernel_size][conv2d_15_kernel_size][conv2d_15_kernel_channels][conv2d_15_kernel_num];
    static int conv2d_15_bias[conv2d_15_kernel_num];
    for (int k = 0; k < conv2d_15_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_15_kernel_size; i++){
            for(int j = 0; j < conv2d_15_kernel_size; j++){
                for(int n = 0; n < conv2d_15_kernel_channels; n++){
                    conv2d_15_kernel[i][j][n][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_15_kernel[i][j][n][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_15_kernel_num; k++){
        conv2d_15_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_15_bias[k]);
    }

    //Perform conv2d_15
    for(int k = 0; k < conv2d_15_kernel_num; k++){
        for(int i = 0; i < conv2d_15_output_size; i++){
            for(int j = 0; j < conv2d_15_output_size; j++){
                conv2d_15_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_15_kernel_size; l++){
                    for(int m = 0; m < conv2d_15_kernel_size; m++){
                        for(int n = 0; n < conv2d_15_kernel_channels; n++){
                            conv2d_15_output[i][j][k] += add_10_output[i + l][j + m][n] * conv2d_15_kernel[l][m][n][k];
                        }
                    }
                }
                conv2d_15_output[i][j][k] += conv2d_15_bias[k];
            }
        }
    }
    printf("Size of conv2d_15_output: %d x %d x %d \n", LEN(conv2d_15_output), LEN(conv2d_15_output[0]), LEN(conv2d_15_output[0][0]));
}

void averagepool2d_1(){
    //Perform averagepool2d_1
    static int averagepool2d_1_output_channels = 960;
    for(int k = 0; k < averagepool2d_1_output_channels; k++){
        for(int i = 0; i < averagepool2d_1_output_size; i++){
            for(int j = 0; j < averagepool2d_1_output_size; j++){
                averagepool2d_1_output[i][j][k] = 0;
                for(int l = 0; l < averagepool2d_1_kernel_size; l++){
                    for(int m = 0; m < averagepool2d_1_kernel_size; m++){
                        averagepool2d_1_output[i][j][k] += conv2d_15_output[i + l][j + m][k];
                    }
                }
                averagepool2d_1_output[i][j][k] /= averagepool2d_1_kernel_size * averagepool2d_1_kernel_size;
            }
        }
    }
    printf("Size of averagepool2d_1_output: %d x %d x %d \n", LEN(averagepool2d_1_output), LEN(averagepool2d_1_output[0]), LEN(averagepool2d_1_output[0][0]));
}

void conv2d_16(){
    static int conv2d_16_kernel[conv2d_16_kernel_size][conv2d_16_kernel_size][conv2d_16_kernel_channels][conv2d_16_kernel_num];
    static int conv2d_16_bias[conv2d_16_kernel_num];
    for (int k = 0; k < conv2d_16_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_16_kernel_size; i++){
            for(int j = 0; j < conv2d_16_kernel_size; j++){
                for(int n = 0; n < conv2d_16_kernel_channels; n++){
                    conv2d_16_kernel[i][j][n][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_16_kernel[i][j][n][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_16_kernel_num; k++){
        conv2d_16_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_16_bias[k]);
    }

    //Perform conv2d_16
    for(int k = 0; k < conv2d_16_kernel_num; k++){
        for(int i = 0; i < conv2d_16_output_size; i++){
            for(int j = 0; j < conv2d_16_output_size; j++){
                conv2d_16_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_16_kernel_size; l++){
                    for(int m = 0; m < conv2d_16_kernel_size; m++){
                        for(int n = 0; n < conv2d_16_kernel_channels; n++){
                            conv2d_16_output[i][j][k] += averagepool2d_1_output[i + l][j + m][n] * conv2d_16_kernel[l][m][n][k];
                        }
                    }
                }
                conv2d_16_output[i][j][k] += conv2d_16_bias[k];
            }
        }
    }
    printf("Size of conv2d_16_output: %d x %d x %d \n", LEN(conv2d_16_output), LEN(conv2d_16_output[0]), LEN(conv2d_16_output[0][0]));
} 

void averagepool2d_2(){
    //Perform averagepool2d_2
    static int averagepool2d_2_output_channels = 1280;
    for(int k = 0; k < averagepool2d_2_output_channels; k++){
        for(int i = 0; i < averagepool2d_2_output_size; i++){
            for(int j = 0; j < averagepool2d_2_output_size; j++){
                averagepool2d_2_output[i][j][k] = 0;
                for(int l = 0; l < averagepool2d_2_kernel_size; l++){
                    for(int m = 0; m < averagepool2d_2_kernel_size; m++){
                        averagepool2d_2_output[i][j][k] += conv2d_16_output[i + l][j + m][k];
                    }
                }
                averagepool2d_2_output[i][j][k] /= averagepool2d_2_kernel_size * averagepool2d_2_kernel_size;
            }
        }
    }
    printf("Size of averagepool2d_2_output: %d x %d x %d \n", LEN(averagepool2d_2_output), LEN(averagepool2d_2_output[0]), LEN(averagepool2d_2_output[0][0]));
}

void conv2d_17(){
    static int conv2d_17_kernel[conv2d_17_kernel_size][conv2d_17_kernel_size][conv2d_17_kernel_channels][conv2d_17_kernel_num];
    static int conv2d_17_bias[conv2d_17_kernel_num];
    for (int k = 0; k < conv2d_17_kernel_num; k++){
        // printf("Kernel %d \n", k);
        for(int i = 0; i < conv2d_17_kernel_size; i++){
            for(int j = 0; j < conv2d_17_kernel_size; j++){
                for(int n = 0; n < conv2d_17_kernel_channels; n++){
                    conv2d_17_kernel[i][j][n][k] = rand() % 5 - 2;
                    // printf("%f ", conv2d_17_kernel[i][j][n][k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //Initialize bias
    for(int k = 0; k < conv2d_17_kernel_num; k++){
        conv2d_17_bias[k] = rand() % 5 - 2;
        // printf("%f ", conv2d_17_bias[k]);
    }

    //Perform conv2d_17
    for(int k = 0; k < conv2d_17_kernel_num; k++){
        for(int i = 0; i < conv2d_17_output_size; i++){
            for(int j = 0; j < conv2d_17_output_size; j++){
                conv2d_17_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_17_kernel_size; l++){
                    for(int m = 0; m < conv2d_17_kernel_size; m++){
                        for(int n = 0; n < conv2d_17_kernel_channels; n++){
                            conv2d_17_output[i][j][k] += averagepool2d_2_output[i + l][j + m][n] * conv2d_17_kernel[l][m][n][k];
                        }
                    }
                }
                conv2d_17_output[i][j][k] += conv2d_17_bias[k];
            }
        }
    }
    printf("Size of conv2d_17_output: %d x %d x %d \n", LEN(conv2d_17_output), LEN(conv2d_17_output[0]), LEN(conv2d_17_output[0][0]));
}

void reshape(){
    //Perform reshape
    for(int i = 0; i < conv2d_17_output_size; i++){
        for(int j = 0; j < conv2d_17_output_size; j++){
            for(int k = 0; k < conv2d_17_kernel_num; k++){
                reshape_output[i * conv2d_17_output_size * conv2d_17_kernel_num + j * conv2d_17_kernel_num + k] = conv2d_17_output[i][j][k];
            }
        }
    }
    printf("Size of reshape_output: %d \n", LEN(reshape_output));
    // for (int i = 0; i < LEN(reshape_output); i++){
    //     printf("reshape_output[%d]: %d \n", i, reshape_output[i]);
    // }
}

void softmax(){
    //Perform softmax
    static int sum = 0;
    for(int i = 0; i < conv2d_17_kernel_num; i++){
        sum += exp(reshape_output[i]);
    }
    for(int i = 0; i < conv2d_17_kernel_num; i++){
        softmax_output[i] = exp(reshape_output[i]) / sum;
    }
    printf("Size of softmax_output: %d \n", LEN(softmax_output));
    // for (int i = 0; i < LEN(softmax_output); i++){
    //     printf("softmax_output[%d]: %d \n", i, softmax_output[i]);
    // }

    //Print top 5 classes
    static int max = 0;
    static int max_index = 0;
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < conv2d_17_kernel_num; j++){
            if(softmax_output[j] > max){
                max = softmax_output[j];
                max_index = j;
            }
        }
        printf("Class %d: %d \n", max_index, max);
        softmax_output[max_index] = 0;
        max = 0;
    }
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
    //Added start
    conv2d_8_bonus();
    depthwiseconv2d_8_bonus();
    pointwiseconv2d_8_bonus();
    //Added stop
    add_5();
    conv2d_8();
    depthwiseconv2d_8();
    pointwiseconv2d_8();
    add_6();
    conv2d_9();
    depthwiseconv2d_9();
    pointwiseconv2d_9();
    add_7();
    conv2d_10();
    depthwiseconv2d_10();
    pointwiseconv2d_10();
    conv2d_11();
    depthwiseconv2d_11();
    pointwiseconv2d_11();
    add_8();
    conv2d_12();
    depthwiseconv2d_12();
    pointwiseconv2d_12();
    conv2d_13();
    depthwiseconv2d_13();
    pointwiseconv2d_13();
    add_9();
    conv2d_14();
    depthwiseconv2d_14();
    pointwiseconv2d_14();
    add_10();
    conv2d_15();
    averagepool2d_1();
    conv2d_16();
    averagepool2d_2();
    conv2d_17();
    reshape();
    softmax();
    return 0;
}
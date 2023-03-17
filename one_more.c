#include <stdio.h>

#define IN_HEIGHT 5
#define IN_WIDTH 5
#define IN_CHANNELS 3
#define DW_FILTER_SIZE 2
#define NUM_FILTERS 2
#define PW_FILTER_SIZE 1
#define OUT_HEIGHT IN_HEIGHT
#define OUT_WIDTH IN_WIDTH
#define OUT_CHANNELS NUM_FILTERS

float in[IN_HEIGHT][IN_WIDTH][IN_CHANNELS] = {{{0}}};
float dw_filter[DW_FILTER_SIZE][DW_FILTER_SIZE][IN_CHANNELS][NUM_FILTERS] = {{{{0}}}};
float pw_filter[PW_FILTER_SIZE][PW_FILTER_SIZE][IN_CHANNELS*NUM_FILTERS][OUT_CHANNELS] = {{{{0}}}};
float out[OUT_HEIGHT][OUT_WIDTH][OUT_CHANNELS] = {{{0}}};

void depthwise_conv2d(float in[IN_HEIGHT][IN_WIDTH][IN_CHANNELS], 
                      float dw_filter[DW_FILTER_SIZE][DW_FILTER_SIZE][IN_CHANNELS][NUM_FILTERS], 
                      float out[OUT_HEIGHT][OUT_WIDTH][OUT_CHANNELS])
{
    int i, j, k, l, m, n;

    // Depthwise convolution
    for (i = 0; i < OUT_HEIGHT; i++) {
        for (j = 0; j < OUT_WIDTH; j++) {
            for (k = 0; k < IN_CHANNELS; k++) {
                for (l = 0; l < DW_FILTER_SIZE; l++) {
                    for (m = 0; m < DW_FILTER_SIZE; m++) {
                        for (n = 0; n < NUM_FILTERS; n++) {
                            out[i][j][k*NUM_FILTERS+n] += 
                                in[i+l][j+m][k] * dw_filter[l][m][k][n];
                        }
                    }
                }
            }
        }
    }

    // Pointwise convolution
    for (i = 0; i < OUT_HEIGHT; i++) {
        for (j = 0; j < OUT_WIDTH; j++) {
            for (k = 0; k < OUT_CHANNELS; k++) {
                for (l = 0; l < IN_CHANNELS*NUM_FILTERS; l++) {
                    out[i][j][k] += in[i][j][l] * pw_filter[0][0][l][k];
                }
            }
        }
    }
}

int main()
{
    // Initialize input, depthwise filter, and pointwise filter
    // ...

    // Perform depthwise separable convolution 2D
    depthwise_conv2d(in, dw_filter, out);

    // Print output
    int i, j, k;
    for (i = 0; i < OUT_HEIGHT; i++) {
        for (j = 0; j < OUT_WIDTH; j++) {
            for (k = 0; k < OUT_CHANNELS; k++) {
                printf("%f ", out[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}
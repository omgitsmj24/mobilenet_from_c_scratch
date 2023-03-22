#include "padding.h"
#include <stdio.h>
#include <stdlib.h>

void padding2d(float input[5][5][3], float output[7][7][3], int input_height, int input_width, 
                int input_channels, int pad_height, int pad_width, int pad_value){
    int output_height = input_height + 2*pad_height;
    int output_width = input_width + 2*pad_width;
    for (int i = 0; i < output_height; i++) {
        for (int j = 0; j < output_width; j++) {
            for (int k = 0; k < input_channels; k++) {
                if (i < pad_height || i >= output_height - pad_height || j < pad_width || j >= output_width - pad_width) {
                    output[i][j][k] = pad_value;
                } else {
                    output[i][j][k] = input[i - pad_height][j - pad_width][k];
                }
            }
        }
    }
}
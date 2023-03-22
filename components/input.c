#include "input.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void init_input(float input[5][5][3], int input_size, int intput_channels) {
    // Initialize input tensor with random values
    for (int d = 0; d < intput_channels; d++) {
        for (int i = 0 ; i < input_size; i++) {
            for (int j = 0; j < input_size; j++) {
                input[i][j][d] = rand() % 10 - 2;
            }
        }
    }
}

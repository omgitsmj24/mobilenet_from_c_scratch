#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 224
#define HEIGHT 224
#define CHANNELS 3

void inputlayer() {
    float input[WIDTH][HEIGHT][CHANNELS];

    // Initialize input layer

    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int k = 0; k < CHANNELS; k++) {
                input[i][j][k] = (float)rand() / (float)RAND_MAX;
            }
        }
    }
    printf("Input layer: \n");
    printf("Channel 1: %f\n", input[0][0][0]);
    printf("Channel 2: %f\n", input[0][0][1]);
    printf("Channel 3: %f\n", input[0][0][2]);
}
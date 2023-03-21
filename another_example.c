#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float* init_input_layer(int width, int height, int depth, int padding) {
    // allocate memory for input layer without padding
    float *input_layer = (float*)malloc(width * height * depth * sizeof(float));

    // initialize input layer with random values between 0 and 1
    srand(time(NULL));
    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                input_layer[d * width * height + i * height + j] = ((float) rand() / (RAND_MAX));
            }
        }
    }

    // allocate memory for input layer with padding
    int padded_width = width + 2*padding;
    int padded_height = height + 2*padding;
    float *padded_input_layer = (float*)malloc(padded_width * padded_height * depth * sizeof(float));

    // initialize padded input layer with zeros
    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < padded_width; i++) {
            for (int j = 0; j < padded_height; j++) {
                padded_input_layer[d * padded_width * padded_height + i * padded_height + j] = 0.0f;
            }
        }
    }

    // copy input layer to padded input layer
    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                padded_input_layer[d * padded_width * padded_height + (i + padding) * padded_height + (j + padding)] = input_layer[d * width * height + i * height + j];
            }
        }
    }

    // free memory for input layer without padding
    free(input_layer);

    // return pointer to padded input layer
    return padded_input_layer;
}

int main() {
    int width = 5;
    int height = 5;
    int depth = 3;
    int padding = 1;

    // initialize input layer with random values and padding
    float* padded_input_layer = init_input_layer(width, height, depth, padding);

    // print padded input layer
    printf("Padded input layer:\n");
    for (int d = 0; d < depth; d++) {
        printf("Channel %d:\n", d);
        for (int i = 0; i < width + 2*padding; i++) {
            for (int j = 0; j < height + 2*padding; j++) {
                printf("%f ", padded_input_layer[d * (width + 2*padding) * (height + 2*padding) + i * (height + 2*padding) + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // free memory for padded input layer
    free(padded_input_layer);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to add zero padding to a 2D input array
// 'input' is the input array to be padded
// 'input_rows' and 'input_cols' are the dimensions of the input array
// 'pad_rows' and 'pad_cols' are the number of rows and columns to be added as padding
// 'pad_top', 'pad_left', 'pad_bottom', and 'pad_right' specify the position of the padding
// Returns a new 2D array with the zero-padded input
float** zero_pad(float** input, int input_rows, int input_cols, int pad_rows, int pad_cols, int pad_top, int pad_left, int pad_bottom, int pad_right) {
    int padded_rows = input_rows + pad_top + pad_bottom;
    int padded_cols = input_cols + pad_left + pad_right;
    float** padded_input = (float**) malloc(padded_rows * sizeof(float*));
    for (int i = 0; i < padded_rows; i++) {
        padded_input[i] = (float*) malloc(padded_cols * sizeof(float));
        memset(padded_input[i], 0, padded_cols * sizeof(float));
    }
    for (int i = pad_top; i < padded_rows - pad_bottom; i++) {
        for (int j = pad_left; j < padded_cols - pad_right; j++) {
            padded_input[i][j] = input[i - pad_top][j - pad_left];
        }
    }
    return padded_input;
}

// Example usage
int main() {
    float** input = (float**) malloc(3 * sizeof(float*));
    for (int i = 0; i < 3; i++) {
        input[i] = (float*) malloc(3 * sizeof(float));
        for (int j = 0; j < 3; j++) {
            input[i][j] = i + j;
        }
    }
    float** padded_input = zero_pad(input, 3, 3, 1, 1, 1, 1, 1, 1);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", padded_input[i][j]);
        }
        printf("\n");
    }
    return 0;
}

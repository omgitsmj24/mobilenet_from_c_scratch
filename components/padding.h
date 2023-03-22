#ifndef PADDING_DOT_H    /* This is an "include guard" */
#define PADDING_DOT_H    /* prevents the file from being included twice. */
                     /* Including a header file twice causes all kinds */
                     /* of interesting problems.*/
                     
void padding2d(float input[5][5][3], float output[7][7][3], int input_height, int input_width, 
                int input_channels, int pad_height, int pad_width, int pad_value);

#endif /* PADDING_DOT_H */
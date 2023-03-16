#ifndef DW_DOT_H    /* This is an "include guard" */
#define DW_DOT_H    /* prevents the file from being included twice. */
                     /* Including a header file twice causes all kinds */
                     /* of interesting problems.*/

/**
 * This is a function declaration.
 * It tells the compiler that the function exists somewhere.
 */

float depthwise_conv(float *input, float *output, float *kernel, int input_h,
                    int input_w, int kernel_h, int kernel_w);

#endif /* DW_DOT_H */
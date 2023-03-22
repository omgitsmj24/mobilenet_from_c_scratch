#ifndef CV2D_DOT_H    /* This is an "include guard" */
#define CV2D_DOT_H    /* prevents the file from being included twice. */
                     /* Including a header file twice causes all kinds */
                     /* of interesting problems.*/
                     
void conv2d(float ***input, int input_size, int input_channels, float ****kernel, int kernel_size, int num_kernels, int kernel_depth, 
            float ***output, float *bias, int stride, int padding);

#endif /* CV2D_DOT_H */
void add_2(){

    // Calculate the input and output new scale
    const float add_2_input_1_scale = 0.5;
    const int add_2_input_1_zero_point = pw3_output_zero_point;
    const float add_2_input_2_scale = pw2_output_scale / (2 * pw3_output_scale);
    const int add_2_input_2_zero_point = pw3_output_zero_point;
    const float add_2_output_scale_new = 2 * pw3_output_scale / add_2_output_scale;

    // Normalized fraction and exponent the add_2_input_1_scale
    float add_2_input_1_scale_fraction;
    int add_2_input_1_scale_exponent;
    frexp_function(add_2_input_1_scale, &add_2_input_1_scale_fraction, &add_2_input_1_scale_exponent);
    int add_2_input_1_scale_fraction_int32 = add_2_input_1_scale_fraction * (1ll << 31);
    printf("add_2_input_1_scale = %f = %d * 2^%d\n", add_2_input_1_scale, add_2_input_1_scale_fraction_int32, add_2_input_1_scale_exponent);

    // Normalized fraction and exponent the add_2_input_2_scale
    float add_2_input_2_scale_fraction;
    int add_2_input_2_scale_exponent;
    frexp_function(add_2_input_2_scale, &add_2_input_2_scale_fraction, &add_2_input_2_scale_exponent);
    int add_2_input_2_scale_fraction_int32 = add_2_input_2_scale_fraction * (1ll << 31);
    printf("add_2_input_2_scale = %f = %d * 2^%d\n", add_2_input_2_scale, add_2_input_2_scale_fraction_int32, add_2_input_2_scale_exponent);

    // Normalized fraction and exponent the add_2_output_scale_new
    float add_2_output_scale_new_fraction;
    int add_2_output_scale_new_exponent;
    frexp_function(add_2_output_scale_new / (1 << 20), &add_2_output_scale_new_fraction, &add_2_output_scale_new_exponent);
    int add_2_output_scale_new_fraction_int32 = add_2_output_scale_new_fraction * (1ll << 31);
    printf("add_2_output_scale_new = %f = %d * 2^%d\n", add_2_output_scale_new, add_2_output_scale_new_fraction_int32, add_2_output_scale_new_exponent);

    // Define size of add_2_input_1, add_2_input_2 and add_2_output_1_32
    int32_t add_2_input_1[add_2_output_size][add_2_output_size][add_2_output_channels];
    int32_t add_2_input_2[add_2_output_size][add_2_output_size][add_2_output_channels];
    int32_t add_2_output_1_32[add_2_output_size][add_2_output_size][add_2_output_channels];
    
    // Copy value of previous layers into add_2_input_1 and add_2_input_2
    for (int i = 0; i < add_2_output_size; i++) {
        for (int j = 0; j < add_2_output_size; j++) {
            for (int k = 0; k < add_2_output_channels; k++) {
                add_2_input_1[i][j][k] = pw1_output[i][j][k];
                add_2_input_2[i][j][k] = conv2d_1_output[i][j][k];
            }   
        }
    }
    printf("add_2_input_1[%d][%d][%d] = %d\n", 0, 0, 0, add_2_input_1[0][0][0]);
    printf("add_2_input_1[%d][%d][%d] = %d\n", 1, 0, 0, add_2_input_1[1][0][0]);

    printf("add_2_input_2[%d][%d][%d] = %d\n", 0, 0, 0, add_2_input_2[0][0][0]);
    printf("add_2_input_2[%d][%d][%d] = %d\n", 1, 0, 0, add_2_input_2[1][0][0]);

    // Perform add_2
    for (int i = 0; i < add_2_output_size; i++) {
        for (int j = 0; j < add_2_output_size; j++) {
            for (int k = 0; k < add_2_output_channels; k++) {

                // Add zero point
                add_2_input_1[i][j][k] = add_2_input_1[i][j][k] + add_2_input_1_zero_point;
                add_2_input_2[i][j][k] = add_2_input_2[i][j][k] + add_2_input_2_zero_point;

                // Left shift the input
                add_2_input_1[i][j][k] = add_2_input_1[i][j][k] * (1 << 20);
                add_2_input_2[i][j][k] = add_2_input_2[i][j][k] * (1 << 20);

                // Fixed point multiplier
                add_2_input_1[i][j][k] = fixed_point_multipilier(add_2_input_1[i][j][k], add_2_input_1_scale_fraction_int32, add_2_input_1_scale_exponent);
                add_2_input_2[i][j][k] = fixed_point_multipilier(add_2_input_2[i][j][k], add_2_input_2_scale_fraction_int32, add_2_input_2_scale_exponent);
                add_2_output_1_32[i][j][k] = add_2_input_1[i][j][k] + add_2_input_2[i][j][k];

                // Fixed point multiplier
                add_2_output_1_32[i][j][k] = fixed_point_multipilier(add_2_output_1_32[i][j][k], add_2_output_scale_new_fraction_int32, add_2_output_scale_new_exponent);

                // Add zero point to add_2_output
                add_2_output_1_32[i][j][k] = add_2_output_1_32[i][j][k] - add_2_output_zero_point;
                
                // Saturate the output
                add_2_output[i][j][k] = saturate(add_2_output_1_32[i][j][k]);
            }   
        }
    }
    printf("First element of add_2_output_1_32: %d\n", add_2_output_1_32[0][0][0]);
    printf("Second element of add_2_output_1_32: %d\n", add_2_output_1_32[1][0][0]);
    printf("First element of add_2_output: %d\n", add_2_output[0][0][0]);
    printf("Second element of add_2_output: %d\n", add_2_output[1][0][0]);
    printf("Size of add_2_output: %d x %d x %d \n\n", LEN(add_2_output), LEN(add_2_output[0]), LEN(add_2_output[0][0]));
}
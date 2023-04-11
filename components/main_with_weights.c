#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include "external/weights.h"
#include "external/input.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr)[0]))

//Input 224x224x3
#define input_size 224
#define input_channels 3

#define input_scale 0.007874015718698502
#define input_zero_point 128

//Conv2d_1 112x112x16
#define conv2d_1_weights_size 3
#define conv2d_1_weights_channels 3
#define conv2d_1_weights_num 16
#define conv2d_1_weights_stride 2
#define conv2d_1_weights_padding (1.0/2.0)
#define conv2d_1_weights_scale 0.14435869455337524 
#define conv2d_1_weights_zero_point -99

#define conv2d_1_bias_scale 0.0011366826947778463 
#define conv2d_1_bias_zero_point 0

#define conv2d_1_output_size 112
#define conv2d_1_output_scale 0.6599291563034058 
#define conv2d_1_output_zero_point 0

uint8_t conv2d_1_output[112][112][16];

//dw1 112x112x16
#define dw1_weights_size 3
#define dw1_weights_channels 16
#define dw1_weights_num 1
#define dw1_weights_stride 1
#define dw1_weights_padding 1
#define dw1_weights_scale 0.6798206567764282 
#define dw1_weights_zero_point -47

#define dw1_bias_scale 0.44863349199295044 
#define dw1_bias_zero_point 0

#define dw1_output_size 112
#define dw1_output_scale 0.7595527172088623 
#define dw1_output_zero_point 0

uint8_t dw1_output[112][112][16];

//pw1 112x112x16
#define pw1_weights_size 1
#define pw1_weights_channels 16
#define pw1_weights_num 16
#define pw1_weights_stride 1
#define pw1_weights_padding 0
#define pw1_weights_scale 0.0181131511926651 
#define pw1_weights_zero_point -121

#define pw1_bias_scale 0.01375789288431406 
#define pw1_bias_zero_point 0

#define pw1_output_size 112
#define pw1_output_scale 2.7218637466430664 
#define pw1_output_zero_point -127

uint8_t pw1_output[112][112][16];

//Add_1 112x112x16
#define add_1_output_size 112
#define add_1_output_channels 16
#define add_1_output_scale 2.807014226913452 
#define add_1_output_zero_point -119

uint8_t add_1_output[112][112][16];

//Conv2d_2 112x112x64
#define conv2d_2_weights_size 1
#define conv2d_2_weights_channels 16
#define conv2d_2_weights_num 64
#define conv2d_2_weights_stride 1
#define conv2d_2_weights_padding 0
#define conv2d_2_weights_scale 0.004174188245087862 
#define conv2d_2_weights_zero_point -142

#define conv2d_2_bias_scale 0.01171700470149517 
#define conv2d_2_bias_zero_point 0

#define conv2d_2_output_size 112
#define conv2d_2_output_scale 0.8431710600852966 
#define conv2d_2_output_zero_point 0

uint8_t conv2d_2_output[112][112][64];

//dw2 56x56x64
#define dw2_weights_size 3
#define dw2_weights_channels 64
#define dw2_weights_num 1
#define dw2_weights_stride 2
#define dw2_weights_padding (1.0/2.0)
#define dw2_weights_scale 0.03459775447845459 
#define dw2_weights_zero_point -109

#define dw2_bias_scale 0.02917182631790638 
#define dw2_bias_zero_point 0

#define dw2_output_size 56
#define dw2_output_scale 0.660646378993988 
#define dw2_output_zero_point 0

uint8_t dw2_output[56][56][64];

//pw2 56x56x24
#define pw2_weights_size 1
#define pw2_weights_channels 64
#define pw2_weights_num 24
#define pw2_weights_stride 1
#define pw2_weights_padding 0
#define pw2_weights_scale 0.012059011496603489 
#define pw2_weights_zero_point -125

#define pw2_bias_scale 0.007966741919517517 
#define pw2_bias_zero_point 0

#define pw2_output_size 56
#define pw2_output_scale 1.0728049278259277 
#define pw2_output_zero_point -131

uint8_t pw2_output[56][56][24];

//conv2d_3 56x56x72
#define conv2d_3_weights_size 1
#define conv2d_3_weights_channels 24
#define conv2d_3_weights_num 72
#define conv2d_3_weights_stride 1
#define conv2d_3_weights_padding 0
#define conv2d_3_weights_scale 0.0045080468989908695 
#define conv2d_3_weights_zero_point -138

#define conv2d_3_bias_scale 0.004836254753172398 
#define conv2d_3_bias_zero_point 0

#define conv2d_3_output_size 56
#define conv2d_3_output_scale 0.2640864849090576 
#define conv2d_3_output_zero_point 0

uint8_t conv2d_3_output[56][56][72];

//dw3 56x56x72
#define dw3_weights_size 3
#define dw3_weights_channels 72
#define dw3_weights_num 1
#define dw3_weights_stride 1
#define dw3_weights_padding 1
#define dw3_weights_scale 0.03428956866264343 
#define dw3_weights_zero_point -96

#define dw3_bias_scale 0.009055412374436855 
#define dw3_bias_zero_point 0

#define dw3_output_size 56
#define dw3_output_scale 0.26957252621650696 
#define dw3_output_zero_point 0

uint8_t dw3_output[56][56][72];

//pw3 56x56x24
#define pw3_weights_size 1
#define pw3_weights_channels 72
#define pw3_weights_num 24
#define pw3_weights_stride 1
#define pw3_weights_padding 0
#define pw3_weights_scale 0.02138252928853035 
#define pw3_weights_zero_point -135

#define pw3_bias_scale 0.005764142610132694 
#define pw3_bias_zero_point 0

#define pw3_output_size 56
#define pw3_output_scale 1.1904277801513672 
#define pw3_output_zero_point -131

uint8_t pw3_output[56][56][24];

//Add_2 56x56x24
#define add_2_output_size 56
#define add_2_output_channels 24
#define add_2_output_scale 1.1451550722122192 
#define add_2_output_zero_point -127
uint8_t add_2_output[56][56][24];

//Conv2d_4 56x56x72
#define conv2d_4_weights_size 1
#define conv2d_4_weights_channels 24
#define conv2d_4_weights_num 72
#define conv2d_4_weights_stride 1
#define conv2d_4_weights_padding 0
#define conv2d_4_weights_scale 0.0025988302659243345 
#define conv2d_4_weights_zero_point -130

#define conv2d_4_bias_scale 0.002976063871756196 
#define conv2d_4_bias_zero_point 0

#define conv2d_4_output_size 56
#define conv2d_4_output_scale 0.2561696767807007 
#define conv2d_4_output_zero_point 0
uint8_t conv2d_4_output[56][56][72];

//dw4 28x28x72
#define dw4_weights_size 3
#define dw4_weights_channels 72
#define dw4_weights_num 1
#define dw4_weights_stride 2
#define dw4_weights_padding (1.0/2.0)
#define dw4_weights_scale 0.03551362082362175 
#define dw4_weights_zero_point -151

#define dw4_bias_scale 0.009097511880099773 
#define dw4_bias_zero_point 0

#define dw4_output_size 28
#define dw4_output_scale 0.23036204278469086 
#define dw4_output_zero_point 0

uint8_t dw4_output[28][28][72];

//pw4 28x28x40
#define pw4_weights_size 1
#define pw4_weights_channels 72
#define pw4_weights_num 40
#define pw4_weights_stride 1
#define pw4_weights_padding 0
#define pw4_weights_scale 0.013196593150496483 
#define pw4_weights_zero_point -146

#define pw4_bias_scale 0.003039994277060032 
#define pw4_bias_zero_point 0

#define pw4_output_size 28
#define pw4_output_scale 0.5330838561058044 
#define pw4_output_zero_point -137

uint8_t pw4_output[28][28][40];

//Conv2d_5 28x28x120
#define conv2d_5_weights_size 1
#define conv2d_5_weights_channels 40
#define conv2d_5_weights_num 120
#define conv2d_5_weights_stride 1
#define conv2d_5_weights_padding 0
#define conv2d_5_weights_scale 0.0030650347471237183 
#define conv2d_5_weights_zero_point -96

#define conv2d_5_bias_scale 0.0016339204739779234 
#define conv2d_5_bias_zero_point 0

#define conv2d_5_output_size 28
#define conv2d_5_output_scale 0.13088467717170715 
#define conv2d_5_output_zero_point 0

uint8_t conv2d_5_output[28][28][120];

//dw5 28x28x120
#define dw5_weights_size 3
#define dw5_weights_channels 120
#define dw5_weights_num 1
#define dw5_weights_stride 1
#define dw5_weights_padding 1
#define dw5_weights_scale 0.04485065117478371 
#define dw5_weights_zero_point -108

#define dw5_bias_scale 0.005870262626558542 
#define dw5_bias_zero_point 0

#define dw5_output_size 28
#define dw5_output_scale 0.1401514858007431 
#define dw5_output_zero_point 0

uint8_t dw5_output[28][28][120];

//pw5 28x28x40
#define pw5_weights_size 1
#define pw5_weights_channels 120
#define pw5_weights_num 40
#define pw5_weights_stride 1
#define pw5_weights_padding 0
#define pw5_weights_scale 0.011324976570904255 
#define pw5_weights_zero_point -144

#define pw5_bias_scale 0.0015872123185545206 
#define pw5_bias_zero_point 0

#define pw5_output_size 28
#define pw5_output_scale 0.4745807349681854 
#define pw5_output_zero_point -130
uint8_t pw5_output[28][28][40];

//Add_3 28x28x40 
#define add_3_output_size 28
#define add_3_output_channels 40
#define add_3_output_scale 0.6054844260215759 
#define add_3_output_zero_point -135

uint8_t add_3_output[28][28][40];

//Conv2d_6 28x28x120
#define conv2d_6_weights_size 1
#define conv2d_6_weights_channels 40
#define conv2d_6_weights_num 120
#define conv2d_6_weights_stride 1
#define conv2d_6_weights_padding 0
#define conv2d_6_weights_scale 0.002430431079119444 
#define conv2d_6_weights_zero_point -117

#define conv2d_6_bias_scale 0.0014715881552547216 
#define conv2d_6_bias_zero_point 0

#define conv2d_6_output_size 28
#define conv2d_6_output_scale 0.11830147355794907 
#define conv2d_6_output_zero_point 0

uint8_t conv2d_6_output[28][28][120];

//dw6 28x28x120
#define dw6_weights_size 3
#define dw6_weights_channels 120
#define dw6_weights_num 1
#define dw6_weights_stride 1
#define dw6_weights_padding 1
#define dw6_weights_scale 0.061444178223609924 
#define dw6_weights_zero_point -145

#define dw6_bias_scale 0.007268936838954687 
#define dw6_bias_zero_point 0

#define dw6_output_size 28
#define dw6_output_scale 0.15098942816257477 
#define dw6_output_zero_point 0

uint8_t dw6_output[28][28][120];

//pw6 28x28x40
#define pw6_weights_size 1
#define pw6_weights_channels 120
#define pw6_weights_num 40
#define pw6_weights_stride 1
#define pw6_weights_padding 0
#define pw6_weights_scale 0.018437182530760765 
#define pw6_weights_zero_point -91

#define pw6_bias_scale 0.0027838198002427816 
#define pw6_bias_zero_point 0

#define pw6_output_size 28
#define pw6_output_scale 0.6320084929466248 
#define pw6_output_zero_point -123

uint8_t pw6_output[28][28][40];

//Add_4 28x28x40
#define add_4_output_size 28
#define add_4_output_channels 40
#define add_4_output_scale 0.6582030653953552 
#define add_4_output_zero_point -124

uint8_t add_4_output[28][28][40];

//Conv2d_7 28x28x240
#define conv2d_7_weights_size 1
#define conv2d_7_weights_channels 40
#define conv2d_7_weights_num 240
#define conv2d_7_weights_stride 1
#define conv2d_7_weights_padding 0
#define conv2d_7_weights_scale 0.002097062300890684 
#define conv2d_7_weights_zero_point -118

#define conv2d_7_bias_scale 0.001380292815156281 
#define conv2d_7_bias_zero_point 0

#define conv2d_7_output_size 28
#define conv2d_7_output_scale 0.13542133569717407 
#define conv2d_7_output_zero_point 0

uint8_t conv2d_7_output[28][28][240];

//dw7 14x14x240
#define dw7_weights_size 3
#define dw7_weights_channels 240
#define dw7_weights_num 1
#define dw7_weights_stride 2
#define dw7_weights_padding (1.0/2.0)
#define dw7_weights_scale 0.031043726950883865 
#define dw7_weights_zero_point -160

#define dw7_bias_scale 0.00420398311689496 
#define dw7_bias_zero_point 0

#define dw7_output_size 14
#define dw7_output_scale 0.16882877051830292 
#define dw7_output_zero_point 0

uint8_t dw7_output[14][14][240];

//pw7 14x14x80
#define pw7_weights_size 1
#define pw7_weights_channels 240
#define pw7_weights_num 80
#define pw7_weights_stride 1
#define pw7_weights_padding 0
#define pw7_weights_scale 0.00881643034517765 
#define pw7_weights_zero_point -115

#define pw7_bias_scale 0.001488467212766409 
#define pw7_bias_zero_point 0

#define pw7_output_size 14
#define pw7_output_scale 0.3794284760951996 
#define pw7_output_zero_point -131
uint8_t pw7_output[14][14][80];

// START - BIG NOTICE - MISSED

//Conv2d_8 14x14x200
#define conv2d_8_weights_size 1
#define conv2d_8_weights_channels 80
#define conv2d_8_weights_num 200
#define conv2d_8_weights_stride 1
#define conv2d_8_weights_padding 0
#define conv2d_8_weights_scale 0.0013372065732255578  
#define conv2d_8_weights_zero_point -125

#define conv2d_8_bias_scale 0.000507374235894531 
#define conv2d_8_bias_zero_point 0

#define conv2d_8_output_size 14
#define conv2d_8_output_scale 0.065852090716362 
#define conv2d_8_output_zero_point 0
uint8_t conv2d_8_output[14][14][200];

//dw8 14x14x200
#define dw8_weights_size 3
#define dw8_weights_channels 200
#define dw8_weights_num 1
#define dw8_weights_stride 1
#define dw8_weights_padding 1
#define dw8_weights_scale 0.07536425441503525 
#define dw8_weights_zero_point -180

#define dw8_bias_scale 0.004962894134223461 
#define dw8_bias_zero_point 0

#define dw8_output_size 14
#define dw8_output_scale 0.09319761395454407 
#define dw8_output_zero_point 0
uint8_t dw8_output[14][14][200];

//pw8 14x14x80
#define pw8_weights_size 1
#define pw8_weights_channels 200
#define pw8_weights_num 80
#define pw8_weights_stride 1
#define pw8_weights_padding 0
#define pw8_weights_scale 0.02751377783715725 
#define pw8_weights_zero_point -97

#define pw8_bias_scale 0.002564218593761325 
#define pw8_bias_zero_point 0

#define pw8_output_size 14
#define pw8_output_scale 0.39363622665405273 
#define pw8_output_zero_point -165
uint8_t pw8_output[14][14][80];

// STOP - BIG NOTICE - MISSED

//Add_5 14x14x80
#define add_5_output_size 14
#define add_5_output_channels 80
#define add_5_output_scale 0.39226800203323364 
#define add_5_output_zero_point -134
uint8_t add_5_output[14][14][80];

//Conv2d_9 14x14x184
#define conv2d_9_weights_size 1
#define conv2d_9_weights_channels 80
#define conv2d_9_weights_num 184
#define conv2d_9_weights_stride 1
#define conv2d_9_weights_padding 0
#define conv2d_9_weights_scale 0.001317909569479525 
#define conv2d_9_weights_zero_point -124

#define conv2d_9_bias_scale 0.0005169737851247191 
#define conv2d_9_bias_zero_point 0

#define conv2d_9_output_size 14
#define conv2d_9_output_scale 0.065852090716362 
#define conv2d_9_output_zero_point 0

uint8_t conv2d_9_output[14][14][184];

//dw8 14x14x184
#define dw9_weights_size 3
#define dw9_weights_channels 184
#define dw9_weights_num 1
#define dw9_weights_stride 1
#define dw9_weights_padding 1
#define dw9_weights_scale 0.045743152499198914 
#define dw9_weights_zero_point -114

#define dw9_bias_scale 0.0027200994081795216 
#define dw9_bias_zero_point 0

#define dw9_output_size 14
#define dw9_output_scale 0.0917566642165184 
#define dw9_output_zero_point 0

uint8_t dw9_output[14][14][184];

//pw8 14x14x80
#define pw9_weights_size 1
#define pw9_weights_channels 184
#define pw9_weights_num 80
#define pw9_weights_stride 1
#define pw9_weights_padding 0
#define pw9_weights_scale 0.012674453668296337 
#define pw9_weights_zero_point -104

#define pw9_bias_scale 0.0011629655491560698 
#define pw9_bias_zero_point 0

#define pw9_output_size 14
#define pw9_output_scale 0.2396366447210312 
#define pw9_output_zero_point -126

uint8_t pw9_output[14][14][80];

//Add_6 14x14x80
#define add_6_output_size 14
#define add_6_output_channels 80
#define add_6_output_scale 0.4093148112297058 
#define add_6_output_zero_point -135

uint8_t add_6_output[14][14][80];

//conv2d_10 14x14x184
#define conv2d_10_weights_size 1
#define conv2d_10_weights_channels 80
#define conv2d_10_weights_num 184
#define conv2d_10_weights_stride 1
#define conv2d_10_weights_padding 0
#define conv2d_10_weights_scale 0.0015550317475572228 
#define conv2d_10_weights_zero_point -139

#define conv2d_10_bias_scale 0.0006364975124597549 
#define conv2d_10_bias_zero_point 0

#define conv2d_10_output_size 14
#define conv2d_10_output_scale 0.06710842996835709 
#define conv2d_10_output_zero_point 0

uint8_t conv2d_10_output[14][14][184];

//dw10 14x14x184
#define dw10_weights_size 3
#define dw10_weights_channels 184
#define dw10_weights_num 1
#define dw10_weights_stride 1
#define dw10_weights_padding 1
#define dw10_weights_scale 0.057766854763031006 
#define dw10_weights_zero_point -154

#define dw10_bias_scale 0.0038766430225223303 
#define dw10_bias_zero_point 0

#define dw10_output_size 14
#define dw10_output_scale 0.11187095195055008 
#define dw10_output_zero_point 0

uint8_t dw10_output[14][14][184];

//pw10 14x14x80
#define pw10_weights_size 1
#define pw10_weights_channels 184
#define pw10_weights_num 80
#define pw10_weights_stride 1
#define pw10_weights_padding 0
#define pw10_weights_scale 0.010866138152778149 
#define pw10_weights_zero_point -122

#define pw10_bias_scale 0.0012156051816418767 
#define pw10_bias_zero_point 0

#define pw10_output_size 14
#define pw10_output_scale 0.31259796023368835 
#define pw10_output_zero_point -132

uint8_t pw10_output[14][14][80];

//Add_7 14x14x80
#define add_7_output_size 14
#define add_7_output_channels 80
#define add_7_output_scale 0.4275743067264557 
#define add_7_output_zero_point -130

uint8_t add_7_output[14][14][80];

//conv2d_11 14x14x480
#define conv2d_11_weights_size 1
#define conv2d_11_weights_channels 80
#define conv2d_11_weights_num 480
#define conv2d_11_weights_stride 1
#define conv2d_11_weights_padding 0
#define conv2d_11_weights_scale 0.0016883378848433495 
#define conv2d_11_weights_zero_point -105

#define conv2d_11_bias_scale 0.000721889897249639 
#define conv2d_11_bias_zero_point 0

#define conv2d_11_output_size 14
#define conv2d_11_output_scale 0.09170106053352356 
#define conv2d_11_output_zero_point 0

uint8_t conv2d_11_output[14][14][480];

//dw11 14x14x480
#define dw11_weights_size 3
#define dw11_weights_channels 480
#define dw11_weights_num 1
#define dw11_weights_stride 1
#define dw11_weights_padding 1
#define dw11_weights_scale 0.049465809017419815
#define dw11_weights_zero_point -134

#define dw11_bias_scale 0.004536067135632038 
#define dw11_bias_zero_point 0

#define dw11_output_size 14
#define dw11_output_scale 0.15485285222530365 
#define dw11_output_zero_point 0

uint8_t dw11_output[14][14][480];

//pw11 14x14x80
#define pw11_weights_size 1
#define pw11_weights_channels 480
#define pw11_weights_num 112
#define pw11_weights_stride 1
#define pw11_weights_padding 0
#define pw11_weights_scale 0.004192234016954899 
#define pw11_weights_zero_point -133

#define pw11_bias_scale 0.0006491793901659548 
#define pw11_bias_zero_point 0

#define pw11_output_size 14
#define pw11_output_scale 0.328067809343338 
#define pw11_output_zero_point -129

uint8_t pw11_output[14][14][112];

//conv2d_12 14x14x672
#define conv2d_12_weights_size 1
#define conv2d_12_weights_channels 112
#define conv2d_12_weights_num 672
#define conv2d_12_weights_stride 1
#define conv2d_12_weights_padding 0
#define conv2d_12_weights_scale 0.0017239248845726252 
#define conv2d_12_weights_zero_point -112

#define conv2d_12_bias_scale 0.0005655642016790807 
#define conv2d_12_bias_zero_point 0

#define conv2d_12_output_size 14
#define conv2d_12_output_scale 0.07632230222225189 
#define conv2d_12_output_zero_point 0

uint8_t conv2d_12_output[14][14][672];

//dw12 14x14x672
#define dw12_weights_size 3
#define dw12_weights_channels 672
#define dw12_weights_num 1
#define dw12_weights_stride 1
#define dw12_weights_padding 1
#define dw12_weights_scale 0.07369618862867355 
#define dw12_weights_zero_point -143

#define dw12_bias_scale 0.00562466261908412 
#define dw12_bias_zero_point 0

#define dw12_output_size 14
#define dw12_output_scale 0.16535194218158722 
#define dw12_output_zero_point 0

uint8_t dw12_output[14][14][672];

//pw12 14x14x80
#define pw12_weights_size 1
#define pw12_weights_channels 672
#define pw12_weights_num 112
#define pw12_weights_stride 1
#define pw12_weights_padding 0
#define pw12_weights_scale 0.007839914411306381 
#define pw12_weights_zero_point -129

#define pw12_bias_scale 0.0012963450280949473 
#define pw12_bias_zero_point 0

#define pw12_output_size 14
#define pw12_output_scale 0.34701263904571533 
#define pw12_output_zero_point -132

uint8_t pw12_output[14][14][112];

//Add_8 14x14x112
#define add_8_output_size 14
#define add_8_output_channels 112
#define add_8_output_scale 0.38235411047935486 
#define add_8_output_zero_point -133
uint8_t add_8_output[14][14][112];

//conv2d_13 14x14x672
#define conv2d_13_weights_size 1
#define conv2d_13_weights_channels 112
#define conv2d_13_weights_num 672
#define conv2d_13_weights_stride 1
#define conv2d_13_weights_padding 0
#define conv2d_13_weights_scale 0.002139686606824398
#define conv2d_13_weights_zero_point -105

#define conv2d_13_bias_scale 0.0008181180455721915 
#define conv2d_13_bias_zero_point 0

#define conv2d_13_output_size 14
#define conv2d_13_output_scale 0.08312690258026123 
#define conv2d_13_output_zero_point 0
uint8_t conv2d_13_output[14][14][672];

//dw13 7x7x672
#define dw13_weights_size 3
#define dw13_weights_channels 672
#define dw13_weights_num 1
#define dw13_weights_stride 2
#define dw13_weights_padding (1.0/2.0)
#define dw13_weights_scale 0.04658949375152588 
#define dw13_weights_zero_point -94

#define dw13_bias_scale 0.0038728404324501753 
#define dw13_bias_zero_point 0

#define dw13_output_size 7
#define dw13_output_scale 0.20200207829475403  
#define dw13_output_zero_point 0

uint8_t dw13_output[7][7][672];

//pw13 7x7x160
#define pw13_weights_size 1
#define pw13_weights_channels 672
#define pw13_weights_num 160
#define pw13_weights_stride 1
#define pw13_weights_padding 0
#define pw13_weights_scale 0.004779632668942213 
#define pw13_weights_zero_point -121

#define pw13_bias_scale 0.0009654957102611661 
#define pw13_bias_zero_point 0

#define pw13_output_size 7
#define pw13_output_scale 0.3152458965778351 
#define pw13_output_zero_point -122

uint8_t pw13_output[7][7][160];

//conv2d_14 7x7x960
#define conv2d_14_weights_size 1
#define conv2d_14_weights_channels 160
#define conv2d_14_weights_num 960
#define conv2d_14_weights_stride 1
#define conv2d_14_weights_padding 0
#define conv2d_14_weights_scale 0.0020118532702326775  
#define conv2d_14_weights_zero_point -149

#define conv2d_14_bias_scale 0.0006342285196296871 
#define conv2d_14_bias_zero_point 0

#define conv2d_14_output_size 7
#define conv2d_14_output_scale 0.11022955179214478 
#define conv2d_14_output_zero_point 0
uint8_t conv2d_14_output[7][7][960];

//dw14 7x7x960
#define dw14_weights_size 3
#define dw14_weights_channels 960
#define dw14_weights_num 1
#define dw14_weights_stride 1
#define dw14_weights_padding 1
#define dw14_weights_scale 0.11063125729560852 
#define dw14_weights_zero_point -130

#define dw14_bias_scale 0.012194833718240261 
#define dw14_bias_zero_point 0

#define dw14_output_size 7
#define dw14_output_scale 0.1480780392885208 
#define dw14_output_zero_point 0

uint8_t dw14_output[7][7][960];

//pw14 7x7x160
#define pw14_weights_size 1
#define pw14_weights_channels 960
#define pw14_weights_num 160
#define pw14_weights_stride 1
#define pw14_weights_padding 0
#define pw14_weights_scale 0.007899901829659939 
#define pw14_weights_zero_point -146

#define pw14_bias_scale 0.0011698019225150347  
#define pw14_bias_zero_point 0

#define pw14_output_size 7
#define pw14_output_scale 0.258070707321167  
#define pw14_output_zero_point -123

uint8_t pw14_output[7][7][160];

//Add_9 7x7x160
#define add_9_output_size 7
#define add_9_output_channels 160
#define add_9_output_scale 0.3708127737045288 
#define add_9_output_zero_point -125

uint8_t add_9_output[7][7][160];

//conv2d_15 7x7x960
#define conv2d_15_weights_size 1
#define conv2d_15_weights_channels 160
#define conv2d_15_weights_num 960
#define conv2d_15_weights_stride 1
#define conv2d_15_weights_padding 0
#define conv2d_15_scale 0.0013757619308307767 
#define conv2d_15_zero_point -108

#define conv2d_15_bias_scale 0.0005101501010358334
#define conv2d_15_bias_zero_point 0

#define conv2d_15_output_size 7
#define conv2d_15_output_scale 0.079544298350811 
#define conv2d_15_output_zero_point 0

uint8_t conv2d_15_output[7][7][960];

//dw15 7x7x960
#define dw15_weights_size 3
#define dw15_weights_channels 960
#define dw15_weights_num 1
#define dw15_weights_stride 1
#define dw15_weights_padding 1
#define dw15_scale 0.0013757619308307767 
#define dw15_zero_point -108

#define dw15_bias_scale 0.0005101501010358334
#define dw15_bias_zero_point 0

#define dw15_output_size 7
#define dw15_output_scale 0.079544298350811 
#define dw15_output_zero_point 0
uint8_t dw15_output[7][7][960];

//pw15 7x7x160
#define pw15_weights_size 1
#define pw15_weights_channels 960
#define pw15_weights_num 160
#define pw15_weights_stride 1
#define pw15_weights_padding 0
#define pw15_scale 0.019383670762181282 
#define pw15_zero_point -167

#define pw15_bias_scale 0.003119788132607937 
#define pw15_bias_zero_point 0

#define pw15_output_size 7
#define pw15_output_scale 0.38830405473709106 
#define pw15_output_zero_point -121

uint8_t pw15_output[7][7][160];

//Add_10 7x7x160
#define add_10_output_size 7
#define add_10_output_channels 160
#define add_10_output_scale 0.38830405473709106 
#define add_10_output_zero_point -121

uint8_t add_10_output[7][7][160];

//conv2d_16 7x7x960
#define conv2d_16_weights_size 1
#define conv2d_16_weights_channels 160
#define conv2d_16_weights_num 960
#define conv2d_16_weights_stride 1
#define conv2d_16_weights_padding 0
#define conv2d_16_scale 0.003714228980243206  
#define conv2d_16_zero_point -134

#define conv2d_16_bias_scale 0.0018642294453456998 
#define conv2d_16_bias_zero_point 0

#define conv2d_16_output_size 7
#define conv2d_16_output_scale 0.4507608115673065 
#define conv2d_16_output_zero_point 0

uint8_t conv2d_16_output[7][7][960];

//AveragePool2d_1 1x1x960
#define averagepool2d_1_weights_size 7
#define averagepool2d_1_weights_stride 1
#define averagepool2d_1_weights_padding 0
#define averagepool2d_1_output_size 1
uint8_t averagepool2d_1_output[1][1][960];

//conv2d_17 1x1x1280
#define conv2d_17_weights_size 1
#define conv2d_17_weights_channels 960
#define conv2d_17_weights_num 1280
#define conv2d_17_weights_stride 1
#define conv2d_17_weights_padding 0
#define conv2d_17_scale 0.008932989090681076  
#define conv2d_17_zero_point -93

#define conv2d_17_bias_scale 0.004026641603559256 
#define conv2d_17_bias_zero_point 0

#define conv2d_17_output_size 1
#define conv2d_17_output_scale 0.019861046224832535  
#define conv2d_17_output_zero_point 0

uint8_t conv2d_17_output[1][1][1280];

//AveragePool2d_2 1x1x1280
#define averagepool2d_2_weights_size 1
#define averagepool2d_2_weights_stride 1
#define averagepool2d_2_weights_padding 0
#define averagepool2d_2_output_size 1
uint8_t averagepool2d_2_output[1][1][1280];

//conv2d_18 1x1x1001
#define conv2d_18_weights_size 1
#define conv2d_18_weights_channels 1280
#define conv2d_18_weights_num 1001
#define conv2d_18_weights_stride 1
#define conv2d_18_weights_padding 0
#define conv2d_18_scale 0.004494924563914537 
#define conv2d_18_zero_point -153

#define conv2d_18_bias_scale 0.00008927390445023775 
#define conv2d_18_bias_zero_point 0

#define conv2d_18_output_size 1
#define conv2d_18_output_scale 0.08109557628631592 
#define conv2d_18_output_zero_point -72

uint8_t conv2d_18_output[1][1][1001];

//Reshape 1001
uint8_t reshape_output[1001];

//Softmax 1001
uint8_t softmax_output[1001];

void frexp_function(float multiplier, float *fraction, int *exponent) {
    *fraction = frexp(multiplier, &(*exponent));
}

int divide_by_power_of_two(int x, int n) {
    int result = x >> n;  // perform the division
    if (x & ((1 << n) - 1)) {  // check if the LSB is 1
        result |= 1;           // round up if needed
    }
    return result;
}

int32_t fixed_point_multipilier(int32_t output_32, int32_t layer_fraction_32, int8_t layer_exponent){
    // Cast to 64 bit
    int64_t output_64 = (int64_t)output_32;
    int64_t layer_fraction_64 = (int64_t)layer_fraction_32;

    //Fixed point multiplication by the quantized multiplier
    int64_t mul_64 = output_64 * layer_fraction_64;
    int32_t nudge = (mul_64 > 0) ? (1 << 30) : (1 - (1 << 30));
    output_32 = (int32_t)((mul_64 + nudge) / (1ll << 31));

    // rounding half away from zero and arithmetic right shift
    output_32 = divide_by_power_of_two(output_32, (-layer_exponent));
    
    return output_32;
}

// Saturate the output to the range of uint8_t (use for adds layers only)
uint8_t saturate(int32_t output_32) {
    if (output_32 > 255) {
        output_32 = 255;
    } else if (output_32 < 0) {
        output_32 = 0;
    }
    return (uint8_t)output_32;
}

void init_input() {
    //Print first element of input
    printf("Size of input: %d x %d x %d \n", LEN(input), LEN(input[0]), LEN(input[0][0]));
    printf("First element of input: %d\n", input[0][0][0]);
    printf("Second element of input: %d\n", input[1][0][0]);
}

void conv2d_1() {

    const float conv2d_1_multiplier = input_scale * conv2d_1_weights_scale / conv2d_1_output_scale;

    // Normalized fraction and exponent the conv2d_1_multiplier
    float conv2d_1_fraction;
    int conv2d_1_exponent;
    frexp_function(conv2d_1_multiplier, &conv2d_1_fraction, &conv2d_1_exponent);
    int conv2d_1_fraction_int32 = conv2d_1_fraction * (1ll << 31);
    printf("conv2d_1_multiplier = %f = %d * 2^%d\n", conv2d_1_multiplier, conv2d_1_fraction_int32, conv2d_1_exponent);

    // Copy input to padded input
    static const int padded_size = 225;
    static uint8_t input_padded[225][225][input_channels];

    // Initilalize padded input with 0
    for (int k = 0; k < input_channels; k++){
        for (int i = 0; i < padded_size; i++) {
            for (int j = 0; j < padded_size; j++) {
                input_padded[i][j][k] = 0;
            }
        }
    }
    // Copy input to padded input
    for (int k = 0; k < input_channels; k++){
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < input_size; j++) {
                input_padded[i][j][k] = input[i][j][k];
            }
        }
    }

    printf("Size of padded input: %d x %d x %d \n", LEN(input_padded), LEN(input_padded[0]), LEN(input_padded[0][0]));

    int32_t conv2d_output_1_32[conv2d_1_output_size][conv2d_1_output_size][conv2d_1_weights_num];

    printf("First element of conv2d_1_weights: %d\n", conv2d_1_weights[0][0][0][0]);
    printf("Second element of conv2d_1_weights: %d\n", conv2d_1_weights[1][0][0][0]);

    // Compute Conv2d_1
    for (int k = 0; k < conv2d_1_weights_num; k++){
        for (int i = 0; i < conv2d_1_output_size; i++) {
            for (int j = 0; j < conv2d_1_output_size; j++) {
                conv2d_output_1_32[i][j][k] = 0;
                for (int l = 0; l < conv2d_1_weights_channels; l++) {
                    for (int m = 0; m < conv2d_1_weights_size; m++) {
                        for (int n = 0; n < conv2d_1_weights_size; n++) {
                            conv2d_output_1_32[i][j][k] += input_padded[i*conv2d_1_weights_stride + m][j*conv2d_1_weights_stride + n][l] * conv2d_1_weights[m][n][l][k];
                        }
                    }
                }
                conv2d_output_1_32[i][j][k] += conv2d_1_biases[k];
                conv2d_output_1_32[i][j][k] = fixed_point_multipilier(conv2d_output_1_32[i][j][k], conv2d_1_fraction_int32, conv2d_1_exponent);

                conv2d_1_output[i][j][k] = (uint8_t)(conv2d_output_1_32[i][j][k]);

                //Add zero point
                conv2d_1_output[i][j][k] = conv2d_1_output[i][j][k] - conv2d_1_output_zero_point;

                //Saturate
                if (conv2d_1_output[i][j][k] < 0) {
                    conv2d_1_output[i][j][k] = 0;
                } else if (conv2d_1_output[i][j][k] > 255) {
                    conv2d_1_output[i][j][k] = 255;
                }
            }
        }
    }
    printf("First element of conv2d_output_1_32: %d\n", conv2d_output_1_32[0][0][0]);
    printf("Second element of conv2d_output_1_32: %d\n", conv2d_output_1_32[1][0][0]);
    printf("First element of conv2d_1_output: %d\n", conv2d_1_output[0][0][0]);
    printf("Second element of conv2d_1_output: %d\n", conv2d_1_output[1][0][0]);
    printf("Size of conv2d_1_output: %d x %d x %d \n\n", LEN(conv2d_1_output), LEN(conv2d_1_output[0]), LEN(conv2d_1_output[0][0]));
}

void dw1(){

    //Calculate dw1_multiplier
    const float dw1_multiplier = conv2d_1_output_scale * dw1_weights_scale / dw1_output_scale;

    // Normalized fraction and exponent the dw2_multiplier
    float dw1_fraction;
    int dw1_exponent;
    frexp_function(dw1_multiplier, &dw1_fraction, &dw1_exponent);
    int dw1_fraction_int32 = dw1_fraction * (1ll << 31);
    printf("dw1_multiplier = %f = %d * 2^%d\n", dw1_multiplier, dw1_fraction_int32, dw1_exponent);

    // Initialize padded conv2d_1_output as 0 array
    static const int conv2d_1_output_padded_size = 114;
    static uint8_t conv2d_1_output_padded[114][114][conv2d_1_weights_num];
    for (int k = 0; k < dw1_weights_channels; k++){
        for (int i = 0; i < conv2d_1_output_padded_size; i++) {
            for (int j = 0; j < conv2d_1_output_padded_size; j++) {
                conv2d_1_output_padded[i][j][k] = 0;
            }
        }
    }
    // Copy conv2d_1_output to padded conv2d_1_output 
    for (int k = 0; k < dw1_weights_channels; k++){
        for (int i = 1; i < conv2d_1_output_size-1; i++) {
            for (int j = 1; j < conv2d_1_output_size-1; j++) {
                conv2d_1_output_padded[i][j][k] = conv2d_1_output[i][j][k];
            }
        }
    }

    // printf("Size of conv2d_1_output: %d x %d x %d \n", LEN(conv2d_1_output), LEN(conv2d_1_output[0]), LEN(conv2d_1_output[0][0]));
    printf("Size of padded conv2d_1_output: %d x %d x %d \n", LEN(conv2d_1_output_padded), LEN(conv2d_1_output_padded[0]), LEN(conv2d_1_output_padded[0][0]));

    int32_t dw1_output_32[dw1_output_size][dw1_output_size][dw1_weights_channels];

    printf("First element of dw1_weights: %d\n", dw1_weights[0][0][0][0]);
    printf("Second element of dw1_weights: %d\n", dw1_weights[1][0][0][0]);

    // Perform dw1
    for (int k = 0; k < dw1_weights_channels; k++){
        for (int i = 0; i < dw1_output_size; i++) {
            for (int j = 0; j < dw1_output_size; j++) {
                dw1_output_32[i][j][k] = 0;
                for (int l = 0; l < dw1_weights_channels; l++) {
                    for (int m = 0; m < dw1_weights_size; m++) {
                        for (int n = 0; n < dw1_weights_size; n++) {
                            dw1_output_32[i][j][k] += conv2d_1_output_padded[i*dw1_weights_stride + m][j*dw1_weights_stride + n][l] * dw1_weights[m][n][l][k];
                        }
                    }
                }
                dw1_output_32[i][j][k] += dw1_biases[k];
                dw1_output_32[i][j][k] = fixed_point_multipilier(dw1_output_32[i][j][k], dw1_fraction_int32, dw1_exponent);

                dw1_output[i][j][k] = (uint8_t)(dw1_output_32[i][j][k]);

                //Add zero point to dw1_output
                dw1_output[i][j][k] -= dw1_output_zero_point;

                //Saturate
                if (dw1_output[i][j][k] > 255){
                    dw1_output[i][j][k] = 255;
                }
                else if (dw1_output[i][j][k] < 0){
                    dw1_output[i][j][k] = 0;
                }
            }
        }
    }
    printf("First element of dw1_output_1_32: %d\n", dw1_output_32[0][0][0]);
    printf("Second element of dw1_output_1_32: %d\n", dw1_output_32[1][0][0]);
    printf("First element of dw1_output: %d\n", dw1_output[0][0][0]);
    printf("Second element of dw1_output: %d\n", dw1_output[1][0][0]);
    printf("Size of dw1_output: %d x %d x %d \n\n", LEN(dw1_output), LEN(dw1_output[0]), LEN(dw1_output[0][0]));
}

void pw1(){

    // Calculate the pw1_multiplier
    const float pw1_multiplier = dw1_output_scale * pw1_weights_scale / pw1_output_scale;

    // Normalized fraction and exponent the pw1_multiplier
    float pw1_fraction;
    int pw1_exponent;
    frexp_function(pw1_multiplier, &pw1_fraction, &pw1_exponent);
    int pw1_fraction_int32 = pw1_fraction * (1ll << 31);
    printf("pw1_multiplier = %f = %d * 2^%d\n", pw1_multiplier, pw1_fraction_int32, pw1_exponent);

    int32_t pw1_output_32[pw1_output_size][pw1_output_size][pw1_weights_num];

    printf("First element of pw1_weights: %d\n", pw1_weights[0][0][0][0]);
    printf("Second element of pw1_weights: %d\n", pw1_weights[0][0][1][0]);

    // Perform pw1
    for(int k = 0; k < pw1_weights_num; k++){
        for (int i = 0; i < pw1_output_size; i++) {
            for (int j = 0; j < pw1_output_size; j++) {
                pw1_output_32[i][j][k] = 0;
                for (int l = 0; l < pw1_weights_channels; l++) {
                    for (int m = 0; m < pw1_weights_size; m++) {
                        for (int n = 0; n < pw1_weights_size; n++) {
                            pw1_output_32[i][j][k] += dw1_output[i*pw1_weights_stride + m][j*pw1_weights_stride + n][l] * pw1_weights[m][n][l][k];
                        }
                    }
                }
                pw1_output_32[i][j][k] += pw1_biases[k];
                pw1_output_32[i][j][k] = fixed_point_multipilier(pw1_output_32[i][j][k], pw1_fraction_int32, pw1_exponent);
                
                pw1_output[i][j][k] = (uint8_t)pw1_output_32[i][j][k];

                //Add zero point to pw1_output
                pw1_output[i][j][k] -= pw1_output_zero_point;

                //Saturate
                if (pw1_output[i][j][k] < 0) {
                    pw1_output[i][j][k] = 0;
                } else if (pw1_output[i][j][k] > 255) {
                    pw1_output[i][j][k] = 255;
                }
            }
        }
    }
    printf("First element of pw1_output_1_32: %d\n", pw1_output_32[0][0][0]);
    printf("Second element of pw1_output_1_32: %d\n", pw1_output_32[1][0][0]);
    printf("First element of pw1_output: %d\n", pw1_output[0][0][0]);
    printf("Second element of pw1_output: %d\n", pw1_output[1][0][0]);
    printf("Size of pw1_output: %d x %d x %d \n\n", LEN(pw1_output), LEN(pw1_output[0]), LEN(pw1_output[0][0]));
}

void add_1() {

    // Calculate the input and output new scale
    const float add_1_input_1_scale = 0.5;
    const int add_1_input_1_zero_point = pw1_output_zero_point;
    const float add_1_input_2_scale = conv2d_1_output_scale / (2 * pw1_output_scale);
    const int add_1_input_2_zero_point = conv2d_1_output_zero_point;
    const float add_1_output_scale_new = 2 * pw1_output_scale / add_1_output_scale;

    // Normalized fraction and exponent the add_1_input_1_scale
    float add_1_input_1_scale_fraction;
    int add_1_input_1_scale_exponent;
    frexp_function(add_1_input_1_scale, &add_1_input_1_scale_fraction, &add_1_input_1_scale_exponent);
    int add_1_input_1_scale_fraction_int32 = add_1_input_1_scale_fraction * (1ll << 31);
    printf("add_1_input_1_scale = %f = %d * 2^%d\n", add_1_input_1_scale, add_1_input_1_scale_fraction_int32, add_1_input_1_scale_exponent);

    // Normalized fraction and exponent the add_1_input_2_scale
    float add_1_input_2_scale_fraction;
    int add_1_input_2_scale_exponent;
    frexp_function(add_1_input_2_scale, &add_1_input_2_scale_fraction, &add_1_input_2_scale_exponent);
    int add_1_input_2_scale_fraction_int32 = add_1_input_2_scale_fraction * (1ll << 31);
    printf("add_1_input_2_scale = %f = %d * 2^%d\n", add_1_input_2_scale, add_1_input_2_scale_fraction_int32, add_1_input_2_scale_exponent);

    // Normalized fraction and exponent the add_1_output_scale_new
    float add_1_output_scale_new_fraction;
    int add_1_output_scale_new_exponent;
    frexp_function(add_1_output_scale_new / (1 << 20), &add_1_output_scale_new_fraction, &add_1_output_scale_new_exponent);
    int add_1_output_scale_new_fraction_int32 = add_1_output_scale_new_fraction * (1ll << 31);
    printf("add_1_output_scale_new = %f = %d * 2^%d\n", add_1_output_scale_new, add_1_output_scale_new_fraction_int32, add_1_output_scale_new_exponent);

    // Define size of add_1_input_1, add_1_input_2 and add_1_output_1_32
    int32_t add_1_input_1[add_1_output_size][add_1_output_size][add_1_output_channels];
    int32_t add_1_input_2[add_1_output_size][add_1_output_size][add_1_output_channels];
    int32_t add_1_output_1_32[add_1_output_size][add_1_output_size][add_1_output_channels];
    
    // Copy value of previous layers into add_1_input_1 and add_1_input_2
    for (int i = 0; i < add_1_output_size; i++) {
        for (int j = 0; j < add_1_output_size; j++) {
            for (int k = 0; k < add_1_output_channels; k++) {
                add_1_input_1[i][j][k] = pw1_output[i][j][k];
                add_1_input_2[i][j][k] = conv2d_1_output[i][j][k];
            }   
        }
    }
    printf("add_1_input_1[%d][%d][%d] = %d\n", 0, 0, 0, add_1_input_1[0][0][0]);
    printf("add_1_input_1[%d][%d][%d] = %d\n", 1, 0, 0, add_1_input_1[1][0][0]);

    printf("add_1_input_2[%d][%d][%d] = %d\n", 0, 0, 0, add_1_input_2[0][0][0]);
    printf("add_1_input_2[%d][%d][%d] = %d\n", 1, 0, 0, add_1_input_2[1][0][0]);
    // Perform add_1    
    for (int i = 0; i < add_1_output_size; i++) {
        for (int j = 0; j < add_1_output_size; j++) {
            for (int k = 0; k < add_1_output_channels; k++) {

                // Add zero point
                add_1_input_1[i][j][k] = add_1_input_1[i][j][k] + add_1_input_1_zero_point;
                add_1_input_2[i][j][k] = add_1_input_2[i][j][k] + add_1_input_2_zero_point;

                // Left shift the input
                add_1_input_1[i][j][k] = add_1_input_1[i][j][k] * (1 << 20);
                add_1_input_2[i][j][k] = add_1_input_2[i][j][k] * (1 << 20);

                // Fixed point multiplier
                add_1_input_1[i][j][k] = fixed_point_multipilier(add_1_input_1[i][j][k], add_1_input_1_scale_fraction_int32, add_1_input_1_scale_exponent);
                add_1_input_2[i][j][k] = fixed_point_multipilier(add_1_input_2[i][j][k], add_1_input_2_scale_fraction_int32, add_1_input_2_scale_exponent);
                add_1_output_1_32[i][j][k] = add_1_input_1[i][j][k] + add_1_input_2[i][j][k];

                // Fixed point multiplier
                add_1_output_1_32[i][j][k] = fixed_point_multipilier(add_1_output_1_32[i][j][k], add_1_output_scale_new_fraction_int32, add_1_output_scale_new_exponent);

                // Add zero point to add_1_output
                add_1_output_1_32[i][j][k] = add_1_output_1_32[i][j][k] - add_1_output_zero_point;
                
                // Saturate the output
                add_1_output[i][j][k] = saturate(add_1_output_1_32[i][j][k]);
            }   
        }
    }
    printf("First element of add_1_output_1_32: %d\n", add_1_output_1_32[0][0][0]);
    printf("Second element of add_1_output_1_32: %d\n", add_1_output_1_32[1][0][0]);
    printf("First element of add_1_output: %d\n", add_1_output[0][0][0]);
    printf("Second element of add_1_output: %d\n", add_1_output[1][0][0]);
    printf("Size of add_1_output: %d x %d x %d \n\n", LEN(add_1_output), LEN(add_1_output[0]), LEN(add_1_output[0][0]));
}

void conv2d_2(){

    // Calculate the conv2d_2_multiplier
    const float conv2d_2_multiplier = add_1_output_scale * conv2d_2_weights_scale / conv2d_2_output_scale;

    // Normalized fraction and exponent the conv2d_2_multiplier
    float conv2d_2_fraction;
    int conv2d_2_exponent;
    frexp_function(conv2d_2_multiplier, &conv2d_2_fraction, &conv2d_2_exponent);
    int conv2d_2_fraction_int32 = conv2d_2_fraction * (1ll << 31);
    printf("conv2d_2_multiplier = %f = %d * 2^%d\n", conv2d_2_multiplier, conv2d_2_fraction_int32, conv2d_2_exponent);

    int32_t conv2d_2_output_32[conv2d_2_output_size][conv2d_2_output_size][conv2d_2_weights_num];

    printf("First element of conv2d_2_weights: %d\n", conv2d_2_weights[0][0][0][0]);
    printf("Second element of conv2d_2_weights: %d\n", conv2d_2_weights[1][0][0][0]);

    for(int k = 0; k < conv2d_2_weights_num; k++){
        for (int i = 0; i < conv2d_2_output_size; i++) {
            for (int j = 0; j < conv2d_2_output_size; j++) {
                conv2d_2_output_32[i][j][k] = 0;
                for (int l = 0; l < conv2d_2_weights_channels; l++) {
                    for (int m = 0; m < conv2d_2_weights_size; m++) {
                        for (int n = 0; n < conv2d_2_weights_size; n++) {
                            conv2d_2_output_32[i][j][k] += add_1_output[i*conv2d_2_weights_stride + m][j*conv2d_2_weights_stride + n][l] * conv2d_2_weights[m][n][l][k];
                        }
                    }
                }
                conv2d_2_output_32[i][j][k] += conv2d_2_biases[k];
                conv2d_2_output_32[i][j][k] = fixed_point_multipilier(conv2d_2_output_32[i][j][k], conv2d_2_fraction_int32, conv2d_2_exponent);

                // Cast to uint8_t
                conv2d_2_output[i][j][k] = (uint8_t)(conv2d_2_output_32[i][j][k]);

                // Add zero point
                conv2d_2_output[i][j][k] = conv2d_2_output[i][j][k] - conv2d_2_output_zero_point;

                // Saturate
                if (conv2d_2_output[i][j][k] < 0) {
                    conv2d_2_output[i][j][k] = 0;
                } else if (conv2d_2_output[i][j][k] > 255) {
                    conv2d_2_output[i][j][k] = 255;
                }
            }
        }
    }
    printf("First element of conv2d_2_output_32: %d\n", conv2d_2_output_32[0][0][0]);
    printf("Second element of conv2d_2_output_32: %d\n", conv2d_2_output_32[1][0][0]);
    printf("First element of conv2d_2_output: %d\n", conv2d_2_output[0][0][0]);
    printf("Second element of conv2d_2_output: %d\n", conv2d_2_output[1][0][0]);
    printf("Size of conv2d_2_output: %d x %d x %d \n\n", LEN(conv2d_2_output), LEN(conv2d_2_output[0]), LEN(conv2d_2_output[0][0]));
}

void dw2(){

    // Calculate the normalized dw2_multiplier
    const float dw2_multiplier = conv2d_2_output_scale * dw2_weights_scale / dw2_output_scale;

    // Normalized fraction and exponent the dw2_multiplier
    float dw2_fraction;
    int dw2_exponent;
    frexp_function(dw2_multiplier, &dw2_fraction, &dw2_exponent);
    int dw2_fraction_int32 = dw2_fraction * (1ll << 31);
    printf("dw2_multiplier = %f = %d * 2^%d\n", dw2_multiplier, dw2_fraction_int32, dw2_exponent);
    

    // Initialize padded conv2d_2_output as 0 array
    static const int conv2d_2_output_padded_size = 113;
    static int conv2d_2_output_padded[113][113][conv2d_2_weights_num];
    for(int k = 0; k < conv2d_2_weights_num; k++){
        for (int i = 0; i < conv2d_2_output_padded_size; i++) {
            for (int j = 0; j < conv2d_2_output_padded_size; j++) {
                conv2d_2_output_padded[i][j][k] = 0;
            }
        }
    }

    // Copy conv2d_2_output to padded conv2d_2_output
    for(int k = 0; k < conv2d_2_weights_num; k++){
        for (int i = 0; i < conv2d_2_output_size; i++) {
            for (int j = 0; j < conv2d_2_output_size; j++) {
                conv2d_2_output_padded[i][j][k] = conv2d_2_output[i][j][k];
            }
        }
    }

    int32_t dw2_output_32[dw2_output_size][dw2_output_size][dw2_weights_channels];

    printf("First element of dw2_weights: %d\n", dw2_weights[0][0][0][0]);
    printf("Second element of dw2_weights: %d\n", dw2_weights[1][0][0][0]);

    printf("Size of padded conv2d_2_output: %d x %d x %d \n", LEN(conv2d_2_output_padded), LEN(conv2d_2_output_padded[0]), LEN(conv2d_2_output_padded[0][0]));

    // Perform dw2
    for(int k = 0; k < dw2_weights_num; k++){
        for (int i = 0; i < dw2_output_size; i++) {
            for (int j = 0; j < dw2_output_size; j++) {
                for (int l = 0; l < dw2_weights_channels; l++) {
                    dw2_output_32[i][j][l] = 0;
                    for (int m = 0; m < dw2_weights_size; m++) {
                        for (int n = 0; n < dw2_weights_size; n++) {
                            dw2_output_32[i][j][l] += conv2d_2_output_padded[i*dw2_weights_stride + m][j*dw2_weights_stride + n][l] * dw2_weights[m][n][l][k];
                        }
                    }
                    dw2_output_32[i][j][l] += dw2_biases[l];

                    dw2_output_32[i][j][l] = fixed_point_multipilier(dw2_output_32[i][j][l], dw2_fraction_int32, dw2_exponent);

                    // Cast to uint8_t
                    dw2_output[i][j][l] = (uint8_t)dw2_output_32[i][j][l];

                    // Add zero point 
                    dw2_output[i][j][l] = dw2_output[i][j][l] - dw2_output_zero_point;

                    // Saturate
                    if (dw2_output[i][j][l] < 0) {
                        dw2_output[i][j][l] = 0;
                    } else if (dw2_output[i][j][l] > 255) {
                        dw2_output[i][j][l] = 255;
                    }
                }
            }
        }
    }
    printf("First element of dw2_output_32: %d\n", dw2_output_32[0][0][0]);
    printf("Second element of dw2_output_32: %d\n", dw2_output_32[1][0][0]);
    printf("First element of dw2_output: %d\n", dw2_output[0][0][0]);
    printf("Second element of dw2_output: %d\n", dw2_output[1][0][0]);
    printf("Size of dw2_output: %d x %d x %d \n\n", LEN(dw2_output), LEN(dw2_output[0]), LEN(dw2_output[0][0]));
}

void pw2(){

    // Calculate the normalized pw2_multiplier
    const float pw2_multiplier = dw2_output_scale * pw2_weights_scale / pw2_output_scale;

    // Normalized fraction and exponent the pw2_multiplier
    float pw2_fraction;
    int pw2_exponent;
    frexp_function(pw2_multiplier, &pw2_fraction, &pw2_exponent);
    int pw2_fraction_int32 = pw2_fraction * (1ll << 31);
    printf("pw2_multiplier = %f = %d * 2^%d\n", pw2_multiplier, pw2_fraction_int32, pw2_exponent);

    int32_t pw2_output_32[pw2_output_size][pw2_output_size][pw2_weights_num];

    printf("First element of pw2_weights: %d\n", pw2_weights[0][0][0][0]);
    printf("Second element of pw2_weights: %d\n", pw2_weights[1][0][0][0]);

    // Perform pw2
    for(int k = 0; k < pw2_weights_num; k++){
        for (int i = 0; i < pw2_output_size; i++) {
            for (int j = 0; j < pw2_output_size; j++) {
                pw2_output_32[i][j][k] = 0;
                for (int l = 0; l < pw2_weights_channels; l++) {
                    for (int m = 0; m < pw2_weights_size; m++) {
                        for (int n = 0; n < pw2_weights_size; n++) {
                            pw2_output_32[i][j][k] += dw2_output[i*pw2_weights_stride + m][j*pw2_weights_stride + n][l] * pw2_weights[m][n][l][k];
                        }
                    }
                }
                pw2_output_32[i][j][k] += pw2_biases[k];

                pw2_output_32[i][j][k] = fixed_point_multipilier(pw2_output_32[i][j][k], pw2_fraction_int32, pw2_exponent);

                // Cast to uint8_t
                pw2_output[i][j][k] = (uint8_t)pw2_output_32[i][j][k];

                // Add zero point 
                pw2_output[i][j][k] = pw2_output[i][j][k] - pw2_output_zero_point;

                // Saturate
                if (pw2_output[i][j][k] < 0) {
                    pw2_output[i][j][k] = 0;
                } else if (pw2_output[i][j][k] > 255) {
                    pw2_output[i][j][k] = 255;
                }

            }
        }
    }
    printf("First element of pw2_output_32: %d\n", pw2_output_32[0][0][0]);
    printf("Second element of pw2_output_32: %d\n", pw2_output_32[1][0][0]);
    printf("First element of pw2_output: %d\n", pw2_output[0][0][0]);
    printf("Second element of pw2_output: %d\n", pw2_output[1][0][0]);
    printf("Size of pw2_output: %d x %d x %d \n\n", LEN(pw2_output), LEN(pw2_output[0]), LEN(pw2_output[0][0]));
}

void conv2d_3(){

    // Calculate the normalized conv2d_3_multiplier
    const float conv2d_3_multiplier = pw2_output_scale * conv2d_3_weights_scale / conv2d_3_output_scale;

    // Normalized fraction and exponent the conv2d_3_multiplier
    float conv2d_3_fraction;
    int conv2d_3_exponent;
    frexp_function(conv2d_3_multiplier, &conv2d_3_fraction, &conv2d_3_exponent);
    int conv2d_3_fraction_int32 = conv2d_3_fraction * (1ll << 31);
    printf("pw2_multiplier = %f = %d * 2^%d\n", conv2d_3_multiplier, conv2d_3_fraction_int32, conv2d_3_exponent);

    int32_t conv2d_3_output_32[conv2d_3_output_size][conv2d_3_output_size][conv2d_3_weights_num];

    printf("First element of conv2d_3_weights: %d\n", conv2d_3_weights[0][0][0][0]);
    printf("Second element of conv2d_3_weights: %d\n", conv2d_3_weights[0][0][1][0]);

    // Perform conv2d_3
    for(int k = 0; k < conv2d_3_weights_num; k++){
        for (int i = 0; i < conv2d_3_output_size; i++) {
            for (int j = 0; j < conv2d_3_output_size; j++) {
                conv2d_3_output_32[i][j][k] = 0;
                for (int l = 0; l < conv2d_3_weights_channels; l++) {
                    for (int m = 0; m < conv2d_3_weights_size; m++) {
                        for (int n = 0; n < conv2d_3_weights_size; n++) {
                            conv2d_3_output_32[i][j][k] += pw2_output[i*conv2d_3_weights_stride + m][j*conv2d_3_weights_stride + n][l] * conv2d_3_weights[m][n][l][k];
                        }
                    }
                }
                conv2d_3_output[i][j][k] += conv2d_3_biases[k];

                conv2d_3_output_32[i][j][k] = fixed_point_multipilier(conv2d_3_output_32[i][j][k], conv2d_3_fraction_int32, conv2d_3_exponent);

                // Cast to uint8_t
                conv2d_3_output[i][j][k] = (uint8_t)conv2d_3_output_32[i][j][k];

                // Add zero point 
                conv2d_3_output[i][j][k] = conv2d_3_output[i][j][k] - conv2d_3_output_zero_point;

                // Saturate
                if (conv2d_3_output[i][j][k] < 0) {
                    conv2d_3_output[i][j][k] = 0;
                } else if (conv2d_3_output[i][j][k] > 255) {
                    conv2d_3_output[i][j][k] = 255;
                }
            }
        }
    }
    printf("First element of conv2d_3_output_32: %d\n", conv2d_3_output_32[0][0][0]);
    printf("Second element of conv2d_3_output_32: %d\n", conv2d_3_output_32[1][0][0]);
    printf("First element of conv2d_3_output: %d\n", conv2d_3_output[0][0][0]);
    printf("Second element of conv2d_3_output: %d\n", conv2d_3_output[1][0][0]);
    printf("Size of conv2d_3_output: %d x %d x %d \n\n", LEN(conv2d_3_output), LEN(conv2d_3_output[0]), LEN(conv2d_3_output[0][0]));
}

void dw3(){

    // Calculate the normalized dw3_multiplier
    const float dw3_multiplier = conv2d_3_output_scale * dw3_weights_scale / dw3_output_scale;

    // Normalized fraction and exponent the dw3_multiplier
    float dw3_fraction;
    int dw3_exponent;
    frexp_function(dw3_multiplier, &dw3_fraction, &dw3_exponent);
    int dw3_fraction_int32 = dw3_fraction * (1ll << 31);
    printf("pw2_multiplier = %f = %d * 2^%d\n", dw3_multiplier, dw3_fraction_int32, dw3_exponent);

    int32_t dw3_output_32[dw3_output_size][dw3_output_size][dw3_weights_channels];

    printf("First element of dw3_weights: %d\n", dw3_weights[0][0][0][0]);
    printf("Second element of dw3_weights: %d\n", dw3_weights[1][0][0][0]);

    // Initialize padded conv2d_3_output with zero value
    static const int padded_conv2d_3_output_size = 58;
    static int padded_conv2d_3_output[58][58][conv2d_3_weights_num];
    for (int i = 0; i < padded_conv2d_3_output_size; i++) {
        for (int j = 0; j < padded_conv2d_3_output_size; j++) {
            for (int k = 0; k < conv2d_3_weights_num; k++) {
                padded_conv2d_3_output[i][j][k] = 0;
                }
            }
        }

    // Copy conv2d_3_output to padded_conv2d_3_output
    for (int i = 0; i < conv2d_3_output_size; i++) {
        for (int j = 0; j < conv2d_3_output_size; j++) {
            for (int k = 0; k < conv2d_3_weights_num; k++) {
                padded_conv2d_3_output[i][j][k] = conv2d_3_output[i][j][k];
            }
        }
    }

    printf("Size of padded_conv2d_3_output: %d x %d x %d \n", LEN(padded_conv2d_3_output), LEN(padded_conv2d_3_output[0]), LEN(padded_conv2d_3_output[0][0]));

    // Perform dw3
    for (int k = 0; k < dw3_weights_channels; k++) {
        for (int i = 0; i < dw3_output_size; i++) {
            for (int j = 0; j < dw3_output_size; j++) {
                dw3_output_32[i][j][k] = 0;
                for (int l = 0; l < dw3_weights_size; l++) {
                    for (int m = 0; m < dw3_weights_size; m++) {
                        dw3_output_32[i][j][k] += padded_conv2d_3_output[i*dw3_weights_stride + l][j*dw3_weights_stride + m][k] * dw3_weights[l][m][k][1];
                    }
                }
                dw3_output_32[i][j][k] += dw3_biases[k];

                dw3_output_32[i][j][k] = fixed_point_multipilier(dw3_output_32[i][j][k], dw3_fraction_int32, dw3_exponent);

                // Cast to uint8_t
                dw3_output[i][j][k] = (uint8_t)dw3_output_32[i][j][k];

                // Add zero point 
                dw3_output[i][j][k] = dw3_output[i][j][k] - dw3_output_zero_point;

                // Saturate
                if (dw3_output[i][j][k] < 0) {
                    dw3_output[i][j][k] = 0;
                } else if (dw3_output[i][j][k] > 255) {
                    dw3_output[i][j][k] = 255;
                }
            }
        }
    }
    printf("First element of dw3_output_32: %d\n", dw3_output_32[0][0][0]);
    printf("Second element of dw3_output_32: %d\n", dw3_output_32[1][0][0]);
    printf("First element of dw3_output: %d\n", dw3_output[0][0][0]);
    printf("Second element of dw3_output: %d\n", dw3_output[1][0][0]);
    printf("Size of dw3_output: %d x %d x %d \n\n", LEN(dw3_output), LEN(dw3_output[0]), LEN(dw3_output[0][0]));
}

void pw3(){

    // Calculate the normalized pw3_multiplier
    const float pw3_multiplier = dw3_output_scale * pw3_weights_scale / pw3_output_scale;

    // Normalized fraction and exponent the pw3_multiplier
    float pw3_fraction;
    int pw3_exponent;
    frexp_function(pw3_multiplier, &pw3_fraction, &pw3_exponent);
    int pw3_fraction_int32 = pw3_fraction * (1ll << 31);
    printf("pw2_multiplier = %f = %d * 2^%d\n", pw3_multiplier, pw3_fraction_int32, pw3_exponent);

    int32_t pw3_output_32[pw3_output_size][pw3_output_size][pw3_weights_num];

    printf("First element of dw3_weights: %d\n", pw3_weights[0][0][0][0]);
    printf("Second element of dw3_weights: %d\n", pw3_weights[0][0][0][1]);

    //Perform pw3
    for(int k = 0; k < pw3_weights_num; k++){
        for (int i = 0; i < pw3_output_size; i++) {
            for (int j = 0; j < pw3_output_size; j++) {
                pw3_output_32[i][j][k] = 0;
                for (int l = 0; l < pw3_weights_channels; l++) {
                    for (int m = 0; m < pw3_weights_size; m++) {
                        for (int n = 0; n < pw3_weights_size; n++) {
                            pw3_output_32[i][j][k] += dw3_output[i*pw3_weights_stride + m][j*pw3_weights_stride + n][l] * pw3_weights[m][n][l][k];
                        }
                    }
                }
                pw3_output_32[i][j][k] += pw3_biases[k];

                pw3_output_32[i][j][k] = fixed_point_multipilier(pw3_output_32[i][j][k], pw3_fraction_int32, pw3_exponent);

                // Cast to uint8_t
                pw3_output[i][j][k] = (uint8_t)pw3_output_32[i][j][k];

                // Add zero point 
                pw3_output[i][j][k] = pw3_output[i][j][k] - pw3_output_zero_point;

                // Saturate
                if (pw3_output[i][j][k] < 0) {
                    pw3_output[i][j][k] = 0;
                } else if (pw3_output[i][j][k] > 255) {
                    pw3_output[i][j][k] = 255;
                }
            }
        }
    }
    printf("First element of pw3_output_32: %d\n", pw3_output_32[0][0][0]);
    printf("Second element of pw3_output_32: %d\n", pw3_output_32[1][0][0]);
    printf("First element of dw3_output: %d\n", pw3_output[0][0][0]);
    printf("Second element of dw3_output: %d\n", pw3_output[1][0][0]);
    printf("Size of pw3_output: %d x %d x %d \n\n", LEN(pw3_output), LEN(pw3_output[0]), LEN(pw3_output[0][0]));
}

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

void conv2d_4(){
    
    // Calculate the normalized conv2d_4_multiplier
    const float conv2d_4_multiplier = add_2_output_scale * conv2d_4_weights_scale / conv2d_4_output_scale;

    // Normalized fraction and exponent the conv2d_4_multiplier
    float conv2d_4_fraction;
    int conv2d_4_exponent;
    frexp_function(conv2d_4_multiplier, &conv2d_4_fraction, &conv2d_4_exponent);
    int conv2d_4_fraction_int32 = conv2d_4_fraction * (1ll << 31);
    printf("pw2_multiplier = %f = %d * 2^%d\n", conv2d_4_multiplier, conv2d_4_fraction_int32, conv2d_4_exponent);

    int32_t conv2d_4_output_32[conv2d_4_output_size][conv2d_4_output_size][conv2d_4_weights_num];

    printf("First element of conv2d_4_weights: %d\n", conv2d_4_weights[0][0][0][0]);
    printf("Second element of conv2d_4_weights: %d\n", conv2d_4_weights[0][0][0][1]);

    //Perform conv2d_4
    for(int k = 0; k < conv2d_4_weights_num; k++){
        for (int i = 0; i < conv2d_4_output_size; i++) {
            for (int j = 0; j < conv2d_4_output_size; j++) {
                conv2d_4_output_32[i][j][k] = 0;
                for (int l = 0; l < conv2d_4_weights_channels; l++) {
                    for (int m = 0; m < conv2d_4_weights_size; m++) {
                        for (int n = 0; n < conv2d_4_weights_size; n++) {
                            conv2d_4_output_32[i][j][k] += add_2_output[i*conv2d_4_weights_stride + m][j*conv2d_4_weights_stride + n][l] * conv2d_4_weights[m][n][l][k];
                        }
                    }
                }
                conv2d_4_output_32[i][j][k] += conv2d_4_biases[k];


                conv2d_4_output_32[i][j][k] = fixed_point_multipilier(conv2d_4_output_32[i][j][k], conv2d_4_fraction_int32, conv2d_4_exponent);

                // Cast to uint8_t
                conv2d_4_output[i][j][k] = (uint8_t)conv2d_4_output_32[i][j][k];

                // Add zero point 
                conv2d_4_output[i][j][k] = conv2d_4_output[i][j][k] - conv2d_4_output_zero_point;

                // Saturate
                if (conv2d_4_output[i][j][k] < 0) {
                    conv2d_4_output[i][j][k] = 0;
                } else if (conv2d_4_output[i][j][k] > 255) {
                    conv2d_4_output[i][j][k] = 255;
                }
                // printf("conv2d_4_output[%d][%d][%d] = %d\n", i, j, k, conv2d_4_output[i][j][k]);
            }
        }
    }
    printf("First element of conv2d_4_output_32: %d\n", conv2d_4_output_32[0][0][0]);
    printf("Second element of conv2d_4_output_32: %d\n", conv2d_4_output_32[1][0][0]);
    printf("First element of conv2d_4_output: %d\n", conv2d_4_output[0][0][0]);
    printf("Second element of conv2d_4_output: %d\n", conv2d_4_output[1][0][0]);
    printf("Size of conv2d_4_output: %d x %d x %d \n\n", LEN(conv2d_4_output), LEN(conv2d_4_output[0]), LEN(conv2d_4_output[0][0]));
}

void dw4(){

    // Calculate the normalized dw4_multiplier
    const float dw4_multiplier = conv2d_4_output_scale * dw4_weights_scale / dw4_output_scale;

    // Normalized fraction and exponent the dw4_multiplier
    float dw4_fraction;
    int dw4_exponent;
    frexp_function(dw4_multiplier, &dw4_fraction, &dw4_exponent);
    int dw4_fraction_int32 = dw4_fraction * (1ll << 31);
    printf("pw2_multiplier = %f = %d * 2^%d\n", dw4_multiplier, dw4_fraction_int32, dw4_exponent);

    int32_t dw4_output_32[dw4_output_size][dw4_output_size][dw4_weights_channels];

    printf("First element of dw4_weights: %d\n", dw4_weights[0][0][0][0]);
    printf("Second element of dw4_weights: %d\n", dw4_weights[0][0][0][1]);

    // Determine padded conv2d_4_output size
    static const int conv2d_4_output_padded_size = 57;
    static uint8_t conv2d_4_output_padded[57][57][conv2d_4_weights_num];
    // Initialize padded conv2d_4_output with zero
    for (int k = 0; k < dw4_weights_channels; k++) {
        for (int i = 0; i < conv2d_4_output_padded_size; i++) {
            for (int j = 0; j < conv2d_4_output_padded_size; j++) {
                conv2d_4_output_padded[i][j][k] = 0;
            }
        }
    }

    // Copy conv2d_4_output to padded conv2d_4_output
    for (int k = 0; k < dw4_weights_channels; k++) {
        for (int i = 0; i < conv2d_4_output_size; i++) {
            for (int j = 0; j < conv2d_4_output_size; j++) {
                conv2d_4_output_padded[i][j][k] = conv2d_4_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_4_output_padded: %d x %d x %d \n", LEN(conv2d_4_output_padded), LEN(conv2d_4_output_padded[0]), LEN(conv2d_4_output_padded[0][0]));

    //Perform dw4
    for (int k = 0; k < dw4_weights_channels; k++) {
        for (int i = 0; i < dw4_output_size; i++) {
            for (int j = 0; j < dw4_output_size; j++) {
                dw4_output_32[i][j][k] = 0;
                for (int l = 0; l < dw4_weights_size; l++) {
                    for (int m = 0; m < dw4_weights_size; m++) {
                        dw4_output_32[i][j][k] += conv2d_4_output_padded[i*dw4_weights_stride + l][j*dw4_weights_stride + m][k] * dw4_weights[l][m][k][1];
                    }
                }
                dw4_output_32[i][j][k] += dw4_biases[k];

                dw4_output_32[i][j][k] = fixed_point_multipilier(dw4_output_32[i][j][k], dw4_fraction_int32, dw4_exponent);

                // Cast to uint8_t
                dw4_output[i][j][k] = (uint8_t)dw4_output_32[i][j][k];

                // Add zero point 
                dw4_output[i][j][k] = dw4_output[i][j][k] - dw4_output_zero_point;

                // Saturate
                if (dw4_output[i][j][k] < 0) {
                    dw4_output[i][j][k] = 0;
                } else if (dw4_output[i][j][k] > 255) {
                    dw4_output[i][j][k] = 255;
                }
            }
        }
    }
    printf("First element of dw4_output_32: %d\n", dw4_output_32[0][0][0]);
    printf("Second element of dw4_output_32: %d\n", dw4_output_32[1][0][0]);
    printf("First element of dw4_output: %d\n", dw4_output[0][0][0]);
    printf("Second element of dw4_output: %d\n", dw4_output[1][0][0]);
    printf("Size of dw4_output: %d x %d x %d \n", LEN(dw4_output), LEN(dw4_output[0]), LEN(dw4_output[0][0]));
}

void pw4(){

    // Perform pw4
    for (int k = 0; k < pw4_weights_num; k++) {
        for (int i = 0; i < pw4_output_size; i++) {
            for (int j = 0; j < pw4_output_size; j++) {
                pw4_output[i][j][k] = 0;
                for (int l = 0; l < pw4_weights_channels; l++) {
                    for (int m = 0; m < pw4_weights_size; m++) {
                        for (int n = 0; n < pw4_weights_size; n++) {
                            pw4_output[i][j][k] += dw4_output[i][j][l] * pw4_weights[m][n][l][k];
                        }
                    }
                }
                pw4_output[i][j][k] += pw4_biases[k];
            }
        }
    }
    printf("First element of pw4_output: %d\n", pw4_output[0][0][0]);
    printf("Size of pw4_output: %d x %d x %d \n", LEN(pw4_output), LEN(pw4_output[0]), LEN(pw4_output[0][0]));
}

void conv2d_5(){

    //Perform conv2d_5
    for(int k = 0; k < conv2d_5_weights_num; k++){
        for(int i = 0; i < conv2d_5_output_size; i++){
            for(int j = 0; j < conv2d_5_output_size; j++){
                conv2d_5_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_5_weights_channels; l++){
                    for(int m = 0; m < conv2d_5_weights_size; m++){
                        for(int n = 0; n < conv2d_5_weights_size; n++){
                            conv2d_5_output[i][j][k] += pw4_output[i*conv2d_5_weights_stride + m][j*conv2d_5_weights_stride + n][l] * conv2d_5_weights[m][n][l][k];
                        }
                    }
                }
                conv2d_5_output[i][j][k] += conv2d_5_biases[k];
            }
        }
    }
    printf("First element of conv2d_5_output: %d\n", conv2d_5_output[0][0][0]);
    printf("Size of conv2d_5_output: %d x %d x %d \n", LEN(conv2d_5_output), LEN(conv2d_5_output[0]), LEN(conv2d_5_output[0][0]));
}

void dw5(){

    // Determine padded conv2d_5_output size
    static const int conv2d_5_padded_output_size = 30;
    static int conv2d_5_padded_output[30][30][conv2d_5_weights_num];
    // Initialize padded conv2d_5_output with 0
    for(int i = 0; i < conv2d_5_padded_output_size; i++){
        for(int j = 0; j < conv2d_5_padded_output_size; j++){
            for(int k = 0; k < dw5_weights_channels; k++){
                conv2d_5_padded_output[i][j][k] = 0;
            }
        }
    }
    // Copy conv2d_5_output to padded conv2d_5_output
    for(int i = 1; i < dw5_output_size - 1; i++){
        for(int j = 1; j < dw5_output_size - 1; j++){
            for(int k = 0; k < dw5_weights_channels; k++){
                conv2d_5_padded_output[i][j][k] = conv2d_5_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_5_padded_output: %d x %d x %d \n", LEN(conv2d_5_padded_output), LEN(conv2d_5_padded_output[0]), LEN(conv2d_5_padded_output[0][0]));

    //Perform dw5
    for (int k = 0; k < dw5_weights_channels; k++){
        for(int i = 0; i < dw5_output_size; i++){
            for(int j = 0; j < dw5_output_size; j++){
                dw5_output[i][j][k] = 0;
                for(int l = 0; l < dw5_weights_channels; l++){
                    for(int m = 0; m < dw5_weights_size; m++){
                        for(int n = 0; n < dw5_weights_size; n++){
                            dw5_output[i][j][l] += conv2d_5_padded_output[i*dw5_weights_stride + m][j*dw5_weights_stride + n][l] * dw5_weights[m][n][l][k];
                        }
                    }
                    dw5_output[i][j][k] += dw5_biases[l];
                }
            }
        }
    }
    printf("First element of dw5_output: %d\n", dw5_output[0][0][0]);
    printf("Size of dw5_output: %d x %d x %d \n", LEN(dw5_output), LEN(dw5_output[0]), LEN(dw5_output[0][0]));
}

void pw5(){

    // Perform pw5
    for(int k = 0; k < pw5_weights_num; k++){
        for(int i = 0; i < dw5_output_size; i++){
            for(int j = 0; j < dw5_output_size; j++){
                pw5_output[i][j][k] = 0;
                for(int l = 0; l < pw5_weights_channels; l++){
                    for(int m = 0; m < pw5_weights_size; m++){
                        for(int n = 0; n < pw5_weights_size; n++){
                            pw5_output[i][j][k] += dw5_output[i*pw5_weights_stride + m][j*pw5_weights_stride + n][l] * pw5_weights[m][n][l][k];
                        }
                    }
                    pw5_output[i][j][k] += pw5_biases[k];
                }
            }
        }
    }
    printf("First element of pw5_output: %d\n", pw5_output[0][0][0]);
    printf("Size of pw5_output: %d x %d x %d \n", LEN(pw5_output), LEN(pw5_output[0]), LEN(pw5_output[0][0]));
}

void add_3(){
    for(int i = 0; i < dw5_output_size; i++){
        for(int j = 0; j < dw5_output_size; j++){
            for(int k = 0; k < dw5_weights_channels; k++){
                add_3_output[i][j][k] = pw5_output[i][j][k] + pw4_output[i][j][k];
            }
        }
    }
    printf("First element of add_3_output: %d\n", add_3_output[0][0][0]);
    printf("Size of add_3_output: %d x %d x %d \n", LEN(add_3_output), LEN(add_3_output[0]), LEN(add_3_output[0][0]));
}

void conv2d_6(){

    // Perform conv2d_6
    for(int k = 0; k < conv2d_6_weights_num; k++){
        for(int i = 0; i < add_3_output_size; i++){
            for(int j = 0; j < add_3_output_size; j++){
                conv2d_6_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_6_weights_channels; l++){
                    for(int m = 0; m < conv2d_6_weights_size; m++){
                        for(int n = 0; n < conv2d_6_weights_size; n++){
                            conv2d_6_output[i][j][k] += add_3_output[i*conv2d_6_weights_stride + m][j*conv2d_6_weights_stride + n][l] * conv2d_6_weights[m][n][l][k];
                        }
                    }
                    conv2d_6_output[i][j][k] += conv2d_6_biases[k];
                }
            }
        }
    }
    printf("First element of conv2d_6_output: %d\n", conv2d_6_output[0][0][0]);
    printf("Size of conv2d_6_output: %d x %d x %d \n", LEN(conv2d_6_output), LEN(conv2d_6_output[0]), LEN(conv2d_6_output[0][0]));
}

void dw6(){

    //Determine the padded conv2d_6_output size
    static const int conv2d_6_output_padded_size = 30;
    static int conv2d_6_output_padded[30][30][dw6_weights_channels];
    //Initialize the padded conv2d_6_output with 0
    for(int i = 0; i < conv2d_6_output_padded_size; i++){
        for(int j = 0; j < conv2d_6_output_padded_size; j++){
            for(int k = 0; k < dw6_weights_channels; k++){
                conv2d_6_output_padded[i][j][k] = 0;
            }
        }
    }

    //Copy the conv2d_6_output to the padded conv2d_6_output
    for(int i = 1; i < conv2d_6_output_size - 1; i++){
        for(int j = 1; j < conv2d_6_output_size - 1; j++){
            for(int k = 1; k < dw6_weights_channels; k++){
                conv2d_6_output_padded[i][j][k] = conv2d_6_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_6_output_padded: %d x %d x %d \n", LEN(conv2d_6_output_padded), LEN(conv2d_6_output_padded[0]), LEN(conv2d_6_output_padded[0][0]));

    //Perform dw6
    for(int k = 0; k < dw6_weights_channels; k++){
        for(int i = 0; i < conv2d_6_output_size; i++){
            for(int j = 0; j < conv2d_6_output_size; j++){
                dw6_output[i][j][k] = 0;
                for(int l = 0; l < dw6_weights_size; l++){
                    for(int m = 0; m < dw6_weights_size; m++){
                        dw6_output[i][j][k] += conv2d_6_output_padded[i*dw6_weights_stride + l][j*dw6_weights_stride + m][k] * dw6_weights[l][m][k][1];
                    }
                }
                dw6_output[i][j][k] += dw6_biases[k];
            }
        }
    }
    printf("First element of dw6_output: %d\n", dw6_output[0][0][0]);
    printf("Size of dw6_output: %d x %d x %d \n", LEN(dw6_output), LEN(dw6_output[0]), LEN(dw6_output[0][0]));
}

void pw6(){

    //Perform pw6
    for(int k = 0; k < pw6_weights_num; k++){
        for(int i = 0; i < conv2d_6_output_size; i++){
            for(int j = 0; j < conv2d_6_output_size; j++){
                pw6_output[i][j][k] = 0;
                for(int l = 0; l < pw6_weights_channels; l++){
                    for(int m = 0; m < pw6_weights_size; m++){
                        for(int n = 0; n < pw6_weights_size; n++){
                            pw6_output[i][j][k] += dw6_output[i*pw6_weights_stride + m][j*pw6_weights_stride + n][l] * pw6_weights[m][n][l][k];
                        }
                    }
                }
                pw6_output[i][j][k] += pw6_biases[k];
            }
        }
    }
    printf("First element of pw6_output: %d\n", pw6_output[0][0][0]);
    printf("Size of pw6_output: %d x %d x %d \n", LEN(pw6_output), LEN(pw6_output[0]), LEN(pw6_output[0][0]));
}

void add_4(){
    for(int i = 0; i < conv2d_6_output_size; i++){
        for(int j = 0; j < conv2d_6_output_size; j++){
            for(int k = 0; k < pw6_weights_num; k++){
                add_4_output[i][j][k] = pw6_output[i][j][k] + add_3_output[i][j][k];
            }
        }
    }
    printf("First element of add_4_output: %d\n", add_4_output[0][0][0]);
    printf("Size of add_4_output: %d x %d x %d \n", LEN(add_4_output), LEN(add_4_output[0]), LEN(add_4_output[0][0]));
}

void conv2d_7(){

    //Perform conv2d_7
    for(int k = 0; k < conv2d_7_weights_num; k++){
        for(int i = 0; i < add_4_output_size; i++){
            for(int j = 0; j < add_4_output_size; j++){
                conv2d_7_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_7_weights_channels; l++){
                    for(int m = 0; m < conv2d_7_weights_size; m++){
                        for(int n = 0; n < conv2d_7_weights_size; n++){
                            conv2d_7_output[i][j][k] += add_4_output[i*conv2d_7_weights_stride + m][j*conv2d_7_weights_stride + n][l] * conv2d_7_weights[m][n][l][k];
                        }
                    }
                }
                conv2d_7_output[i][j][k] += conv2d_7_biases[k];
            }
        }
    }
    printf("First element of conv2d_7_output: %d\n", conv2d_7_output[0][0][0]);
    printf("Size of conv2d_7_output: %d x %d x %d \n", LEN(conv2d_7_output), LEN(conv2d_7_output[0]), LEN(conv2d_7_output[0][0]));
}

void dw7(){

    //Determine the padded conv2d_7_output size
    static const int conv2d_7_output_padded_size = 29;
    static int conv2d_7_output_padded[29][29][dw7_weights_channels];

    //Initialize padded conv2d_7_output with 0
    for(int i = 0; i < conv2d_7_output_padded_size; i++){
        for(int j = 0; j < conv2d_7_output_padded_size; j++){
            for(int k = 0; k < dw7_weights_channels; k++){
                conv2d_7_output_padded[i][j][k] = 0;
            }
        }
    }

    //Copy conv2d_7_output to padded conv2d_7_output
    for(int i = 0; i < conv2d_7_output_size; i++){
        for(int j = 0; j < conv2d_7_output_size; j++){
            for(int k = 0; k < dw7_weights_channels; k++){
                conv2d_7_output_padded[i][j][k] = conv2d_7_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_7_output_padded: %d x %d x %d \n", LEN(conv2d_7_output_padded), LEN(conv2d_7_output_padded[0]), LEN(conv2d_7_output_padded[0][0]));

    //Perform dw7
    for(int k = 0; k < dw7_weights_channels; k++){
        for(int i = 0; i < dw7_output_size; i++){
            for(int j = 0; j < dw7_output_size; j++){
                dw7_output[i][j][k] = 0;
                for(int l = 0; l < dw7_weights_size; l++){
                    for(int m = 0; m < dw7_weights_size; m++){
                        dw7_output[i][j][k] += conv2d_7_output_padded[i*dw7_weights_stride + l][j*dw7_weights_stride + m][k] * dw7_weights[l][m][k][k];
                    }
                }
                dw7_output[i][j][k] += dw7_biases[k];
            }
        }
    }
    printf("First element of dw7_output: %d\n", dw7_output[0][0][0]);
    printf("Size of dw7_output: %d x %d x %d \n", LEN(dw7_output), LEN(dw7_output[0]), LEN(dw7_output[0][0]));
}

void pw7(){

    //Perform pw7
    for(int k = 0; k < pw7_weights_num; k++){
        for(int i = 0; i < pw7_output_size; i++){
            for(int j = 0; j < pw7_output_size; j++){
                pw7_output[i][j][k] = 0;
                for(int l = 0; l < pw7_weights_channels; l++){
                    for(int m = 0; m < pw7_weights_size; m++){
                        for(int n = 0; n < pw7_weights_size; n++){
                            pw7_output[i][j][k] += dw7_output[i][j][l] * pw7_weights[m][n][l][k];
                        }
                    }
                }
                pw7_output[i][j][k] += pw7_biases[k];
            }
        }
    }
    printf("First element of pw7_output: %d\n", pw7_output[0][0][0]);
    printf("Size of pw7_output: %d x %d x %d \n", LEN(pw7_output), LEN(pw7_output[0]), LEN(pw7_output[0][0]));
}

void conv2d_8(){

    //Perform conv2d_8
    for(int k = 0; k < conv2d_8_weights_num; k++){
        for(int i = 0; i < conv2d_8_output_size; i++){
            for(int j = 0; j < conv2d_8_output_size; j++){
                conv2d_8_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_8_weights_channels; l++){
                    for(int m = 0; m < conv2d_8_weights_size; m++){
                        for(int n = 0; n < conv2d_8_weights_size; n++){
                            conv2d_8_output[i][j][k] += pw7_output[i][j][l] * conv2d_8_weights[m][n][l][k];
                        }
                    }
                }
                conv2d_8_output[i][j][k] += conv2d_8_biases[k];
            }
        }
    }
    printf("First element of conv2d_8_output: %d\n", conv2d_8_output[0][0][0]);
    printf("Size of conv2d_8_output: %d x %d x %d \n", LEN(conv2d_8_output), LEN(conv2d_8_output[0]), LEN(conv2d_8_output[0][0]));
}

void dw8(){

    //Determine the padded conv2d_8_output size 
    static const int conv2d_8_padded_output_size = 16;
    static int conv2d_8_padded_output[16][16][dw8_weights_channels];

    //Initialize the padded conv2d_8_output with 0
    for(int i = 0; i < conv2d_8_padded_output_size; i++){
        for(int j = 0; j < conv2d_8_padded_output_size; j++){
            for(int k = 0; k < dw8_weights_channels; k++){
                conv2d_8_padded_output[i][j][k] = 0;
            }
        }
    }

    //Copy the conv2d_8_output to the padded conv2d_8_output
    for(int i = 1; i < dw8_output_size - 1; i++){
        for(int j = 1; j < dw8_output_size - 1; j++){
            for(int k = 0; k < dw8_weights_channels; k++){
                conv2d_8_padded_output[i][j][k] = conv2d_8_output[i][j][k];
            }
        }
    }
    printf("First element of conv2d_8_padded_output: %d\n", conv2d_8_padded_output[0][0][0]);
    printf("Size of conv2d_8_padded_output: %d x %d x %d \n", LEN(conv2d_8_padded_output), LEN(conv2d_8_padded_output[0]), LEN(conv2d_8_padded_output[0][0]));
    
    //Perform dw8
    for(int k = 0; k < dw8_weights_num; k++){
        for(int i = 0; i < dw8_output_size; i++){
            for(int j = 0; j < dw8_output_size; j++){
                dw8_output[i][j][k] = 0;
                for(int l = 0; l < dw8_weights_channels; l++){
                    for(int m = 0; m < dw8_weights_size; m++){
                        for(int n = 0; n < dw8_weights_size; n++){
                            dw8_output[i][j][l] += conv2d_8_padded_output[i + m][j + n][l] * dw8_weights[m][n][l][k];
                        }
                    }
                    dw8_output[i][j][k] += dw8_biases[l];
                }
            }
        }
    }
    printf("First element of dw8_output: %d\n", dw8_output[0][0][0]);
    printf("Size of dw8_output: %d x %d x %d \n", LEN(dw8_output), LEN(dw8_output[0]), LEN(dw8_output[0][0]));
}

void pw8(){

    //Perform pw8
    for(int k = 0; k < pw8_weights_num; k++){
        for(int i = 0; i < pw8_output_size; i++){
            for(int j = 0; j < pw8_output_size; j++){
                pw8_output[i][j][k] = 0;
                for(int l = 0; l < pw8_weights_channels; l++){
                    for(int m = 0; m < pw8_weights_size; m++){
                        for(int n = 0; n < pw8_weights_size; n++){
                            pw8_output[i][j][k] += dw8_output[i + m][j + n][l] * pw8_weights[m][n][l][k];
                        }
                    }
                    pw8_output[i][j][k] += pw8_biases[k];
                }
            }
        }
    }
    printf("First element of pw8_output: %d\n", pw8_output[0][0][0]);
    printf("Size of pw8_output: %d x %d x %d \n", LEN(pw8_output), LEN(pw8_output[0]), LEN(pw8_output[0][0]));
}

void add_5(){
    for(int i = 0; i < add_5_output_size; i++){
        for(int j = 0; j < add_5_output_size; j++){
            for(int k = 0; k < add_5_output_channels; k++){
                add_5_output[i][j][k] = pw7_output[i][j][k] + pw8_output[i][j][k];
            }
        }
    }
    printf("First element of add_5_output: %d\n", add_5_output[0][0][0]);
    printf("Size of add_5_output: %d x %d x %d \n", LEN(add_5_output), LEN(add_5_output[0]), LEN(add_5_output[0][0]));
}

void conv2d_9(){

    //Perform conv2d_9
    for(int k = 0; k < conv2d_9_weights_num; k++){
        for(int i = 0; i < conv2d_9_output_size; i++){
            for(int j = 0; j < conv2d_9_output_size; j++){
                conv2d_9_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_9_weights_channels; l++){
                    for(int m = 0; m < conv2d_9_weights_size; m++){
                        for(int n = 0; n < conv2d_9_weights_size; n++){
                            conv2d_9_output[i][j][k] += add_5_output[i + m][j + n][l] * conv2d_9_weights[m][n][l][k];
                        }
                    }
                }
                conv2d_9_output[i][j][k] += conv2d_9_biases[k];
            }
        }
    }
    printf("First element of conv2d_9_output: %d\n", conv2d_9_output[0][0][0]);
    printf("Size of conv2d_9_output: %d x %d x %d \n", LEN(conv2d_9_output), LEN(conv2d_9_output[0]), LEN(conv2d_9_output[0][0]));
}

void dw9(){

    //Determine padded conv2d_9_output size
    static const int conv2d_9_output_padded_size = 16;
    static int conv2d_9_output_padded[16][16][dw9_weights_channels];

    //Initialize padded conv2d_9_output with 0
    for(int i = 0; i < conv2d_9_output_padded_size; i++){
        for(int j = 0; j < conv2d_9_output_padded_size; j++){
            for(int k = 0; k < dw9_weights_channels; k++){
                conv2d_9_output_padded[i][j][k] = 0;
            }
        }
    }
    //Copy conv2d_9_output to padded conv2d_9_output
    for(int i = 1; i < conv2d_9_output_size - 1; i++){
        for(int j = 1; j < conv2d_9_output_size - 1; j++){
            for(int k = 0; k < dw9_weights_channels; k++){
                conv2d_9_output_padded[i][j][k] = conv2d_9_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_9_output_padded: %d x %d x %d \n", LEN(conv2d_9_output_padded), LEN(conv2d_9_output_padded[0]), LEN(conv2d_9_output_padded[0][0]));

    //Perform dw8
    for(int k = 0; k < dw9_weights_channels; k++){
        for(int i = 0; i < conv2d_9_output_size; i++){
            for(int j = 0; j < conv2d_9_output_size; j++){
                dw9_output[i][j][k] = 0;
                for(int l = 0; l < dw9_weights_size; l++){
                    for(int m = 0; m < dw9_weights_size; m++){
                        dw9_output[i][j][k] += conv2d_9_output_padded[i + l][j + m][k] * dw9_weights[l][m][k][1];
                    }
                }
                dw9_output[i][j][k] += dw9_biases[k];
            }
        }
    }
    printf("First element of dw9_output: %d\n", dw9_output[0][0][0]);
    printf("Size of dw9_output: %d x %d x %d \n", LEN(dw9_output), LEN(dw9_output[0]), LEN(dw9_output[0][0]));
}

void pw9(){

    //Perform pw8
    for(int k = 0; k < pw9_weights_num; k++){
        for(int i = 0; i < conv2d_9_output_size; i++){
            for(int j = 0; j < conv2d_9_output_size; j++){
                pw9_output[i][j][k] = 0;
                for(int l = 0; l < pw9_weights_size; l++){
                    for(int m = 0; m < pw9_weights_size; m++){
                        for(int n = 0; n < pw9_weights_channels; n++){
                            pw9_output[i][j][k] += dw9_output[i + l][j + m][n] * pw9_weights[l][m][n][k];
                        }
                    }
                }
                pw9_output[i][j][k] += pw9_biases[k];
            }
        }
    }
    printf("First element of pw9_output: %d\n", pw9_output[0][0][0]);
    printf("Size of pw9_output: %d x %d x %d \n", LEN(pw9_output), LEN(pw9_output[0]), LEN(pw9_output[0][0]));
}

void add_6(){
    //Perform add_6
    for(int i = 0; i < conv2d_9_output_size; i++){
        for(int j = 0; j < conv2d_9_output_size; j++){
            for(int k = 0; k < pw9_weights_num; k++){
                add_6_output[i][j][k] = pw9_output[i][j][k] + add_5_output[i][j][k];
            }
        }
    }
    printf("First element of add_6_output: %d\n", add_6_output[0][0][0]);
    printf("Size of add_6_output: %d x %d x %d \n", LEN(add_6_output), LEN(add_6_output[0]), LEN(add_6_output[0][0]));
}

void conv2d_10(){

    //Perform conv2d_10
    for(int k = 0; k < conv2d_10_weights_num; k++){
        for(int i = 0; i < conv2d_10_output_size; i++){
            for(int j = 0; j < conv2d_10_output_size; j++){
                conv2d_10_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_10_weights_size; l++){
                    for(int m = 0; m < conv2d_10_weights_size; m++){
                        for(int n = 0; n < conv2d_10_weights_channels; n++){
                            conv2d_10_output[i][j][k] += add_6_output[i + l][j + m][n] * conv2d_10_weights[l][m][n][k];
                        }
                    }
                }
                conv2d_10_output[i][j][k] += conv2d_10_biases[k];
            }
        }
    }
    printf("First element of conv2d_10_output: %d\n", conv2d_10_output[0][0][0]);
    printf("Size of conv2d_10_output: %d x %d x %d \n", LEN(conv2d_10_output), LEN(conv2d_10_output[0]), LEN(conv2d_10_output[0][0]));
}

void dw10(){

    //Determine the padded conv2d_10_output size
    static const int conv2d_10_output_padded_size = 16;
    static int conv2d_10_output_padded[16][16][dw10_weights_channels];

    //Initialize the padded conv2d_10_output with 0
    for(int i = 0; i < conv2d_10_output_padded_size; i++){
        for(int j = 0; j < conv2d_10_output_padded_size; j++){
            for(int k = 0; k < dw10_weights_channels; k++){
                conv2d_10_output_padded[i][j][k] = 0;
            }
        }
    }

    //Copy conv2d_10_output to the padded conv2d_10_output
    for(int i = 1; i < conv2d_10_output_size - 1; i++){
        for(int j = 1; j < conv2d_10_output_size - 1; j++){
            for(int k = 0; k < dw10_weights_channels; k++){
                conv2d_10_output_padded[i][j][k] = conv2d_10_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_10_output_padded: %d x %d x %d \n", LEN(conv2d_10_output_padded), LEN(conv2d_10_output_padded[0]), LEN(conv2d_10_output_padded[0][0]));

    //Perform dw10
    for(int k = 0; k < dw10_weights_channels; k++){
        for(int i = 0; i < dw10_output_size; i++){
            for(int j = 0; j < dw10_output_size; j++){
                dw10_output[i][j][k] = 0;
                for(int l = 0; l < dw10_weights_size; l++){
                    for(int m = 0; m < dw10_weights_size; m++){
                        dw10_output[i][j][k] += conv2d_10_output_padded[i + l][j + m][k] * dw10_weights[l][m][k][1];
                    }
                }
                dw10_output[i][j][k] += dw10_biases[k];
            }
        }
    }
    printf("First element of dw10_output: %d\n", dw10_output[0][0][0]);
    printf("Size of dw10_output: %d x %d x %d \n", LEN(dw10_output), LEN(dw10_output[0]), LEN(dw10_output[0][0]));
}

void pw10(){

    //Perform pw10
    for(int k = 0; k < pw10_weights_num; k++){
        for(int i = 0; i < pw10_output_size; i++){
            for(int j = 0; j < pw10_output_size; j++){
                pw10_output[i][j][k] = 0;
                for(int l = 0; l < pw10_weights_channels; l++){
                    pw10_output[i][j][k] += dw10_output[i][j][l] * pw10_weights[0][0][l][k];
                }
                pw10_output[i][j][k] += pw10_biases[k];
            }
        }
    }
    printf("First element of pw10_output: %d\n", pw10_output[0][0][0]);
    printf("Size of pw10_output: %d x %d x %d \n", LEN(pw10_output), LEN(pw10_output[0]), LEN(pw10_output[0][0]));
}

void add_7(){
    //Perform add_7
    for(int i = 0; i < add_7_output_size; i++){
        for(int j = 0; j < add_7_output_size; j++){
            for(int k = 0; k < add_7_output_channels; k++){
                add_7_output[i][j][k] = pw10_output[i][j][k] + add_6_output[i][j][k];
            }
        }
    }
    printf("First element of add_7_output: %d\n", add_7_output[0][0][0]);
    printf("Size of add_7_output: %d x %d x %d \n", LEN(add_7_output), LEN(add_7_output[0]), LEN(add_7_output[0][0]));
}

void conv2d_11(){

    //Perform conv2d_11
    for(int k = 0; k < conv2d_11_weights_num; k++){
        for(int i = 0; i < conv2d_11_output_size; i++){
            for(int j = 0; j < conv2d_11_output_size; j++){
                conv2d_11_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_11_weights_channels; l++){
                    for(int m = 0; m < conv2d_11_weights_size; m++){
                        for(int n = 0; n < conv2d_11_weights_size; n++){
                            conv2d_11_output[i][j][k] += add_7_output[i + m][j + n][l] * conv2d_11_weights[m][n][l][k];
                        }
                    }
                }
                conv2d_11_output[i][j][k] += conv2d_11_biases[k];
            }
        }
    }
    printf("First element of conv2d_11_output: %d\n", conv2d_11_output[0][0][0]);
    printf("Size of conv2d_11_output: %d x %d x %d \n", LEN(conv2d_11_output), LEN(conv2d_11_output[0]), LEN(conv2d_11_output[0][0]));
}

void dw11(){

    //Determine the padded conv2d_11_output size
    static const int conv2d_11_output_padded_size = 16;
    static int conv2d_11_output_padded[16][16][dw11_weights_channels];

    //Initialize the padded conv2d_11_output with 0
    for(int i = 0; i < conv2d_11_output_padded_size; i++){
        for(int j = 0; j < conv2d_11_output_padded_size; j++){
            for(int k = 0; k < dw11_weights_channels; k++){
                conv2d_11_output_padded[i][j][k] = 0;
            }
        }
    }

    //Copy the conv2d_11_output to the padded conv2d_11_output
    for(int i = 1; i < conv2d_11_output_size - 1; i++){
        for(int j = 1; j < conv2d_11_output_size - 1; j++){
            for(int k = 0; k < dw11_weights_channels; k++){
                conv2d_11_output_padded[i][j][k] = conv2d_11_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_11_output_padded: %d x %d x %d \n", LEN(conv2d_11_output_padded), LEN(conv2d_11_output_padded[0]), LEN(conv2d_11_output_padded[0][0]));

    //Perform dw11
    for(int k = 0; k < dw11_weights_channels; k++){
        for(int i = 0; i < dw11_output_size; i++){
            for(int j = 0; j < dw11_output_size; j++){
                dw11_output[i][j][k] = 0;
                for(int l = 0; l < dw11_weights_size; l++){
                    for(int m = 0; m < dw11_weights_size; m++){
                        dw11_output[i][j][k] += conv2d_11_output_padded[i + l][j + m][k] * dw11_weights[l][m][k][k];
                    }
                }
                dw11_output[i][j][k] += dw11_biases[k];
            }
        }
    }
    printf("First element of dw11_output: %d\n", dw11_output[0][0][0]);
    printf("Size of dw11_output: %d x %d x %d \n", LEN(dw11_output), LEN(dw11_output[0]), LEN(dw11_output[0][0]));
}

void pw11(){

    //Perform pw11
    for(int k = 0; k < pw11_weights_num; k++){
        for(int i = 0; i < pw11_output_size; i++){
            for(int j = 0; j < pw11_output_size; j++){
                pw11_output[i][j][k] = 0;
                for(int l = 0; l < pw11_weights_channels; l++){
                    pw11_output[i][j][k] += dw11_output[i][j][l] * pw11_weights[0][0][l][k];
                }
                pw11_output[i][j][k] += pw11_biases[k];
            }
        }
    }
    printf("First element of pw11_output: %d\n", pw11_output[0][0][0]);
    printf("Size of pw11_output: %d x %d x %d \n", LEN(pw11_output), LEN(pw11_output[0]), LEN(pw11_output[0][0]));
}

void conv2d_12(){

    //Perform conv2d_12
    for(int k = 0; k < conv2d_12_weights_num; k++){
        for(int i = 0; i < conv2d_12_output_size; i++){
            for(int j = 0; j < conv2d_12_output_size; j++){
                conv2d_12_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_12_weights_size; l++){
                    for(int m = 0; m < conv2d_12_weights_size; m++){
                        for(int n = 0; n < conv2d_12_weights_channels; n++){
                            conv2d_12_output[i][j][k] += pw11_output[i + l][j + m][n] * conv2d_12_weights[l][m][n][k];
                        }
                    }
                }
                conv2d_12_output[i][j][k] += conv2d_12_biases[k];
            }
        }
    }
    printf("First element of conv2d_12_output: %d\n", conv2d_12_output[0][0][0]);
    printf("Size of conv2d_12_output: %d x %d x %d \n", LEN(conv2d_12_output), LEN(conv2d_12_output[0]), LEN(conv2d_12_output[0][0]));
}

void dw12(){

    //Determine the padded conv2d_12_output size
    static const int conv2d_12_output_padded_size = 16;
    static int conv2d_12_output_padded[16][16][dw12_weights_channels];

    //Copy conv2d_12_output to conv2d_12_output_padded
    for(int k = 0; k < dw12_weights_channels; k++){
        for(int i = 1; i < conv2d_12_output_size - 1; i++){
            for(int j = 1; j < conv2d_12_output_size - 1; j++){
                conv2d_12_output_padded[i][j][k] = conv2d_12_output[i][j][k];
            }
        }
    }

    printf("Size of conv2d_12_output_padded: %d x %d x %d \n", LEN(conv2d_12_output_padded), LEN(conv2d_12_output_padded[0]), LEN(conv2d_12_output_padded[0][0]));

    //Perform dw12
    for(int k = 0; k < dw12_weights_channels; k++){
        for(int i = 0; i < dw12_output_size; i++){
            for(int j = 0; j < dw12_output_size; j++){
                dw12_output[i][j][k] = 0;
                for(int l = 0; l < dw12_weights_size; l++){
                    for(int m = 0; m < dw12_weights_size; m++){
                        dw12_output[i][j][k] += conv2d_12_output_padded[i + l][j + m][k] * dw12_weights[l][m][k][1];
                    }
                }
                dw12_output[i][j][k] += dw12_biases[k];
            }
        }
    }
    printf("First element of dw12_output: %d\n", dw12_output[0][0][0]);
    printf("Size of dw12_output: %d x %d x %d \n", LEN(dw12_output), LEN(dw12_output[0]), LEN(dw12_output[0][0]));
}

void pw12(){

    //Perform pw12
    for(int k = 0; k < pw12_weights_num; k++){
        for(int i = 0; i < pw12_output_size; i++){
            for(int j = 0; j < pw12_output_size; j++){
                pw12_output[i][j][k] = 0;
                for(int l = 0; l < pw12_weights_size; l++){
                    for(int m = 0; m < pw12_weights_size; m++){
                        for(int n = 0; n < pw12_weights_channels; n++){
                            pw12_output[i][j][k] += dw12_output[i + l][j + m][n] * pw12_weights[l][m][n][k];
                        }
                    }
                }
                pw12_output[i][j][k] += pw12_biases[k];
            }
        }
    }
    printf("First element of pw12_output: %d\n", pw12_output[0][0][0]);
    printf("Size of pw12_output: %d x %d x %d \n", LEN(pw12_output), LEN(pw12_output[0]), LEN(pw12_output[0][0]));
}

void add_8(){
    //Perform add_8
    for(int k = 0; k < add_8_output_channels; k++){
        for(int i = 0; i < add_8_output_size; i++){
            for(int j = 0; j < add_8_output_size; j++){
                add_8_output[i][j][k] = pw12_output[i][j][k] + pw11_output[i][j][k];
            }
        }
    }
    printf("First element of add_8_output: %d\n", add_8_output[0][0][0]);
    printf("Size of add_8_output: %d x %d x %d \n", LEN(add_8_output), LEN(add_8_output[0]), LEN(add_8_output[0][0]));
}

void conv2d_13(){

    //Perform conv2d_13
    for(int k = 0; k < conv2d_13_weights_num; k++){
        for(int i = 0; i < conv2d_13_output_size; i++){
            for(int j = 0; j < conv2d_13_output_size; j++){
                conv2d_13_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_13_weights_size; l++){
                    for(int m = 0; m < conv2d_13_weights_size; m++){
                        for(int n = 0; n < conv2d_13_weights_channels; n++){
                            conv2d_13_output[i][j][k] += add_8_output[i + l][j + m][n] * conv2d_13_weights[l][m][n][k];
                        }
                    }
                }
                conv2d_13_output[i][j][k] += conv2d_13_biases[k];
            }
        }
    }
    printf("First element of conv2d_13_output: %d\n", conv2d_13_output[0][0][0]);
    printf("Size of conv2d_13_output: %d x %d x %d \n", LEN(conv2d_13_output), LEN(conv2d_13_output[0]), LEN(conv2d_13_output[0][0]));
}

void dw13(){

    //Determine padded conv2d_13_output size
    static const int conv2d_13_output_padded_size = 15;
    static int conv2d_13_output_padded[15][15][conv2d_13_weights_num];

    //Initialize conv2d_13_output_padded with 0
    for(int k = 0; k < conv2d_13_weights_num; k++){
        for(int i = 0; i < conv2d_13_output_padded_size; i++){
            for(int j = 0; j < conv2d_13_output_padded_size; j++){
                conv2d_13_output_padded[i][j][k] = 0;
            }
        }
    }
    //Copy conv2d_13_output to conv2d_13_output_padded
    for(int k = 0; k < conv2d_13_weights_num; k++){
        for(int i = 0; i < conv2d_13_output_size; i++){
            for(int j = 0; j < conv2d_13_output_size; j++){
                conv2d_13_output_padded[i][j][k] = conv2d_13_output[i][j][k];
            }
        }
    }
    printf("Size of conv2d_13_output_padded: %d x %d x %d \n", LEN(conv2d_13_output_padded), LEN(conv2d_13_output_padded[0]), LEN(conv2d_13_output_padded[0][0]));
    //Perform dw13
    for(int k = 0; k < dw13_weights_channels; k++){
        for(int i = 0; i < dw13_output_size; i++){
            for(int j = 0; j < dw13_output_size; j++){
                dw13_output[i][j][k] = 0;
                for(int l = 0; l < dw13_weights_size; l++){
                    for(int m = 0; m < dw13_weights_size; m++){
                        dw13_output[i][j][k] += conv2d_13_output_padded[i + l][j + m][k] * dw13_weights[l][m][k][1];
                    }
                }
                dw13_output[i][j][k] += dw13_biases[k];
            }
        }
    }
    printf("First element of dw13_output: %d\n", dw13_output[0][0][0]);
    printf("Size of dw13_output: %d x %d x %d \n", LEN(dw13_output), LEN(dw13_output[0]), LEN(dw13_output[0][0]));
}

void pw13(){

    //Perform pw13
    for(int k = 0; k < pw13_weights_num; k++){
        for(int i = 0; i < pw13_output_size; i++){
            for(int j = 0; j < pw13_output_size; j++){
                pw13_output[i][j][k] = 0;
                for(int l = 0; l < pw13_weights_size; l++){
                    for(int m = 0; m < pw13_weights_size; m++){
                        for(int n = 0; n < pw13_weights_channels; n++){
                            pw13_output[i][j][k] += dw13_output[i + l][j + m][n] * pw13_weights[l][m][n][k];
                        }
                    }
                }
                pw13_output[i][j][k] += pw13_biases[k];
            }
        }
    }
    printf("First element of pw13_output: %d\n", pw13_output[0][0][0]);
    printf("Size of pw13_output: %d x %d x %d \n", LEN(pw13_output), LEN(pw13_output[0]), LEN(pw13_output[0][0]));
}

void conv2d_14(){

    //Perform conv2d_14
    for(int k = 0; k < conv2d_14_weights_num; k++){
        for(int i = 0; i < conv2d_14_output_size; i++){
            for(int j = 0; j < conv2d_14_output_size; j++){
                conv2d_14_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_14_weights_size; l++){
                    for(int m = 0; m < conv2d_14_weights_size; m++){
                        for(int n = 0; n < conv2d_14_weights_channels; n++){
                            conv2d_14_output[i][j][k] += pw13_output[i + l][j + m][n] * conv2d_14_weights[l][m][n][k];
                        }
                    }
                }
                conv2d_14_output[i][j][k] += conv2d_14_biases[k];
            }
        }
    }
    printf("First element of conv2d_14_output: %d\n", conv2d_14_output[0][0][0]);
    printf("Size of conv2d_14_output: %d x %d x %d \n", LEN(conv2d_14_output), LEN(conv2d_14_output[0]), LEN(conv2d_14_output[0][0]));
}

void dw14(){

    //Determine the padded conv2d_14_output size 
    static const int conv2d_14_output_padded_size = 9;
    static int conv2d_14_output_padded[9][9][conv2d_14_weights_num];

    //Initialize the padded conv2d_14_output with 0
    for(int k = 0; k < conv2d_14_weights_num; k++){
        for(int i = 0; i < conv2d_14_output_padded_size; i++){
            for(int j = 0; j < conv2d_14_output_padded_size; j++){
                conv2d_14_output_padded[i][j][k] = 0;
            }
        }
    }
    //Copy the conv2d_14_output to the padded conv2d_14_output
    for(int k = 0; k < conv2d_14_weights_num; k++){
        for(int i = 1; i < conv2d_14_output_size - 1; i++){
            for(int j = 1; j < conv2d_14_output_size - 1; j++){
                conv2d_14_output_padded[i][j][k] = conv2d_14_output[i][j][k];
            }
        }
    }
    printf("Size of conv2d_14_output_padded: %d x %d x %d \n", LEN(conv2d_14_output_padded), LEN(conv2d_14_output_padded[0]), LEN(conv2d_14_output_padded[0][0]));

    //Perform dw14
    for(int k = 0; k < dw14_weights_channels; k++){
        for(int i = 0; i < dw14_output_size; i++){
            for(int j = 0; j < dw14_output_size; j++){
                dw14_output[i][j][k] = 0;
                for(int l = 0; l < dw14_weights_size; l++){
                    for(int m = 0; m < dw14_weights_size; m++){
                        dw14_output[i][j][k] += conv2d_14_output[i + l][j + m][k] * dw14_weights[l][m][k][1];
                    }
                }
                dw14_output[i][j][k] += dw14_biases[k];
            }
        }
    }
    printf("First element of dw14_output: %d\n", dw14_output[0][0][0]);
    printf("Size of dw14_output: %d x %d x %d \n", LEN(dw14_output), LEN(dw14_output[0]), LEN(dw14_output[0][0]));
}

void pw14(){

    //Perform pw14
    for(int k = 0; k < pw14_weights_num; k++){
        for(int i = 0; i < pw14_output_size; i++){
            for(int j = 0; j < pw14_output_size; j++){
                pw14_output[i][j][k] = 0;
                for(int l = 0; l < pw14_weights_size; l++){
                    for(int m = 0; m < pw14_weights_size; m++){
                        for(int n = 0; n < pw14_weights_channels; n++){
                            pw14_output[i][j][k] += dw14_output[i + l][j + m][n] * pw14_weights[l][m][n][k];
                        }
                    }
                }
            }
        }
    }
    printf("First element of pw14_output: %d\n", pw14_output[0][0][0]);
    printf("Size of pw14_output: %d x %d x %d \n", LEN(pw14_output), LEN(pw14_output[0]), LEN(pw14_output[0][0]));
}

void add_9(){
    //Perform add_9
    for(int k = 0; k < add_9_output_channels; k++){
        for(int i = 0; i < add_9_output_size; i++){
            for(int j = 0; j < add_9_output_size; j++){
                add_9_output[i][j][k] = pw14_output[i][j][k] + pw13_output[i][j][k];
            }
        }
    }
    printf("First element of add_9_output: %d\n", add_9_output[0][0][0]);
    printf("Size of add_9_output: %d x %d x %d \n", LEN(add_9_output), LEN(add_9_output[0]), LEN(add_9_output[0][0]));
}

void conv2d_15(){

    //Perform conv2d_15
    for(int k = 0; k < conv2d_15_weights_num; k++){
        for(int i = 0; i < conv2d_15_output_size; i++){
            for(int j = 0; j < conv2d_15_output_size; j++){
                conv2d_15_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_15_weights_size; l++){
                    for(int m = 0; m < conv2d_15_weights_size; m++){
                        for(int n = 0; n < conv2d_15_weights_channels; n++){
                            conv2d_15_output[i][j][k] += add_9_output[i + l][j + m][n] * conv2d_15_weights[l][m][n][k];
                        }
                    }
                }
                conv2d_15_output[i][j][k] += conv2d_15_biases[k];
            }
        }
    }
    printf("First element of conv2d_15_output: %d\n", conv2d_15_output[0][0][0]);
    printf("Size of conv2d_15_output: %d x %d x %d \n", LEN(conv2d_15_output), LEN(conv2d_15_output[0]), LEN(conv2d_15_output[0][0]));
}

void dw15(){

    //Determine the padded conv2d_15_output size
    static const int conv2d_15_output_padded_size = 9;
    static int conv2d_15_output_padded[9][9][conv2d_15_weights_num];

    //Initialize the padded conv2d_15_output with 0
    for(int k = 0; k < conv2d_15_weights_num; k++){
        for(int i = 0; i < conv2d_15_output_padded_size; i++){
            for(int j = 0; j < conv2d_15_output_padded_size; j++){
                conv2d_15_output_padded[i][j][k] = 0;
            }
        }
    }

    //Copy the conv2d_15_output to the padded conv2d_15_output
    for(int k = 0; k < conv2d_15_weights_num; k++){
        for(int i = 1; i < conv2d_15_output_size - 1; i++){
            for(int j = 1; j < conv2d_15_output_size - 1; j++){
                conv2d_15_output_padded[i][j][k] = conv2d_15_output[i][j][k];
            }
        }
    }
    printf("Size of conv2d_15_output_padded: %d x %d x %d \n", LEN(conv2d_15_output_padded), LEN(conv2d_15_output_padded[0]), LEN(conv2d_15_output_padded[0][0]));

    //Perform dw15
    for(int k = 0; k < dw15_weights_channels; k++){
        for(int i = 0; i < dw15_output_size; i++){
            for(int j = 0; j < dw15_output_size; j++){
                dw15_output[i][j][k] = 0;
                for(int l = 0; l < dw15_weights_size; l++){
                    for(int m = 0; m < dw15_weights_size; m++){
                        dw15_output[i][j][k] += conv2d_15_output[i + l][j + m][k] * dw15_weights[l][m][k][1];
                    }
                }
                dw15_output[i][j][k] += dw15_biases[k];
            }
        }
    }
    printf("First element of dw15_output: %d\n", dw15_output[0][0][0]);
    printf("Size of dw15_output: %d x %d x %d \n", LEN(dw15_output), LEN(dw15_output[0]), LEN(dw15_output[0][0]));
}

void pw15(){

    //Perform pw15
    for(int k = 0; k < pw15_weights_num; k++){
        for(int i = 0; i < pw15_output_size; i++){
            for(int j = 0; j < pw15_output_size; j++){
                pw15_output[i][j][k] = 0;
                for(int l = 0; l < pw15_weights_size; l++){
                    for(int m = 0; m < pw15_weights_size; m++){
                        for(int n = 0; n < pw15_weights_channels; n++){
                            pw15_output[i][j][k] += dw15_output[i + l][j + m][n] * pw15_weights[l][m][n][k];
                        }
                    }
                }
                pw15_output[i][j][k] += pw15_biases[k];
            }
        }
    }
    printf("First element of pw15_output: %d\n", pw15_output[0][0][0]);
    printf("Size of pw15_output: %d x %d x %d \n", LEN(pw15_output), LEN(pw15_output[0]), LEN(pw15_output[0][0]));
}

void add_10(){
    //Perform add_10
    for(int k = 0; k < add_10_output_channels; k++){
        for(int i = 0; i < add_10_output_size; i++){
            for(int j = 0; j < add_10_output_size; j++){
                add_10_output[i][j][k] = pw15_output[i][j][k] + add_9_output[i][j][k];
            }
        }
    }
    printf("First element of add_10_output: %d\n", add_10_output[0][0][0]);
    printf("Size of add_10_output: %d x %d x %d \n", LEN(add_10_output), LEN(add_10_output[0]), LEN(add_10_output[0][0]));
}

void conv2d_16(){

    //Perform conv2d_16
    for(int k = 0; k < conv2d_16_weights_num; k++){
        for(int i = 0; i < conv2d_16_output_size; i++){
            for(int j = 0; j < conv2d_16_output_size; j++){
                conv2d_16_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_16_weights_size; l++){
                    for(int m = 0; m < conv2d_16_weights_size; m++){
                        for(int n = 0; n < conv2d_16_weights_channels; n++){
                            conv2d_16_output[i][j][k] += add_10_output[i + l][j + m][n] * conv2d_16_weights[l][m][n][k];
                        }
                    }
                }
                conv2d_16_output[i][j][k] += conv2d_16_biases[k];
            }
        }
    }
    printf("First element of conv2d_16_output: %d\n", conv2d_16_output[0][0][0]);
    printf("Size of conv2d_16_output: %d x %d x %d \n", LEN(conv2d_16_output), LEN(conv2d_16_output[0]), LEN(conv2d_16_output[0][0]));
}

void averagepool2d_1(){
    //Perform averagepool2d_1
    static int averagepool2d_1_output_channels = 960;
    for(int k = 0; k < averagepool2d_1_output_channels; k++){
        for(int i = 0; i < averagepool2d_1_output_size; i++){
            for(int j = 0; j < averagepool2d_1_output_size; j++){
                averagepool2d_1_output[i][j][k] = 0;
                for(int l = 0; l < averagepool2d_1_weights_size; l++){
                    for(int m = 0; m < averagepool2d_1_weights_size; m++){
                        averagepool2d_1_output[i][j][k] += conv2d_16_output[i + l][j + m][k];
                    }
                }
                averagepool2d_1_output[i][j][k] /= averagepool2d_1_weights_size * averagepool2d_1_weights_size;
            }
        }
    }
    printf("First element of averagepool2d_1_output: %d\n", averagepool2d_1_output[0][0][0]);
    printf("Size of averagepool2d_1_output: %d x %d x %d \n", LEN(averagepool2d_1_output), LEN(averagepool2d_1_output[0]), LEN(averagepool2d_1_output[0][0]));
}

void conv2d_17(){

    //Perform conv2d_17
    for(int k = 0; k < conv2d_17_weights_num; k++){
        for(int i = 0; i < conv2d_17_output_size; i++){
            for(int j = 0; j < conv2d_17_output_size; j++){
                conv2d_17_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_17_weights_size; l++){
                    for(int m = 0; m < conv2d_17_weights_size; m++){
                        for(int n = 0; n < conv2d_17_weights_channels; n++){
                            conv2d_17_output[i][j][k] += averagepool2d_1_output[i + l][j + m][n] * conv2d_17_weights[l][m][n][k];
                        }
                    }
                }
                conv2d_17_output[i][j][k] += conv2d_17_biases[k];
            }
        }
    }
    printf("First element of conv2d_17_output: %d\n", conv2d_17_output[0][0][0]);
    printf("Size of conv2d_17_output: %d x %d x %d \n", LEN(conv2d_17_output), LEN(conv2d_17_output[0]), LEN(conv2d_17_output[0][0]));
} 

void averagepool2d_2(){
    //Perform averagepool2d_2
    static int averagepool2d_2_output_channels = 1280;
    for(int k = 0; k < averagepool2d_2_output_channels; k++){
        for(int i = 0; i < averagepool2d_2_output_size; i++){
            for(int j = 0; j < averagepool2d_2_output_size; j++){
                averagepool2d_2_output[i][j][k] = 0;
                for(int l = 0; l < averagepool2d_2_weights_size; l++){
                    for(int m = 0; m < averagepool2d_2_weights_size; m++){
                        averagepool2d_2_output[i][j][k] += conv2d_17_output[i + l][j + m][k];
                    }
                }
                averagepool2d_2_output[i][j][k] /= averagepool2d_2_weights_size * averagepool2d_2_weights_size;
            }
        }
    }
    printf("First element of averagepool2d_2_output: %d\n", averagepool2d_2_output[0][0][0]);
    printf("Size of averagepool2d_2_output: %d x %d x %d \n", LEN(averagepool2d_2_output), LEN(averagepool2d_2_output[0]), LEN(averagepool2d_2_output[0][0]));
}

void conv2d_18(){

    //Perform conv2d_18
    for(int k = 0; k < conv2d_18_weights_num; k++){
        for(int i = 0; i < conv2d_18_output_size; i++){
            for(int j = 0; j < conv2d_18_output_size; j++){
                conv2d_18_output[i][j][k] = 0;
                for(int l = 0; l < conv2d_18_weights_size; l++){
                    for(int m = 0; m < conv2d_18_weights_size; m++){
                        for(int n = 0; n < conv2d_18_weights_channels; n++){
                            conv2d_18_output[i][j][k] += averagepool2d_2_output[i + l][j + m][n] * conv2d_18_weights[l][m][n][k];
                        }
                    }
                }
                conv2d_18_output[i][j][k] += conv2d_18_biases[k];
            }
        }
    }
    printf("First element of conv2d_18_output: %d\n", conv2d_18_output[0][0][0]);
    printf("Size of conv2d_18_output: %d x %d x %d \n", LEN(conv2d_18_output), LEN(conv2d_18_output[0]), LEN(conv2d_18_output[0][0]));
}

void reshape(){
    //Perform reshape
    for(int i = 0; i < conv2d_18_output_size; i++){
        for(int j = 0; j < conv2d_18_output_size; j++){
            for(int k = 0; k < conv2d_18_weights_num; k++){
                reshape_output[i * conv2d_18_output_size * conv2d_18_weights_num + j * conv2d_18_weights_num + k] = conv2d_18_output[i][j][k];
            }
        }
    }
    printf("First element of reshape_output: %d\n", reshape_output[0]);
    printf("Size of reshape_output: %d \n", LEN(reshape_output));
    // for (int i = 0; i < LEN(reshape_output); i++){
    //     printf("reshape_output[%d]: %d \n", i, reshape_output[i]);
    // }
}

void softmax(){
    //Perform softmax
    static int sum = 0;
    for(int i = 0; i < conv2d_18_weights_num; i++){
        sum += exp(reshape_output[i]);
    }
    for(int i = 0; i < conv2d_18_weights_num; i++){
        softmax_output[i] = exp(reshape_output[i]) / sum;
    }
    printf("First element of softmax_output: %d\n", softmax_output[0]);
    printf("Size of softmax_output: %d \n", LEN(softmax_output));
    // for (int i = 0; i < LEN(softmax_output); i++){
    //     printf("softmax_output[%d]: %d \n", i, softmax_output[i]);
    // }

    //Print top 5 classes
    static int max = 0;
    static int max_index = 0;
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < conv2d_18_weights_num; j++){
            if(softmax_output[j] > max){
                max = softmax_output[j];
                max_index = j;
            }
        }
        printf("Class %d: %d \n", max_index, max);
        softmax_output[max_index] = 0;
        max = 0;
    }
}

int main (){
    clock_t start_time = clock();
    init_input();
    conv2d_1();
    dw1();
    pw1();
    add_1();
    conv2d_2();
    dw2();
    pw2();
    conv2d_3();
    dw3();
    pw3();
    add_2();
    conv2d_4();
    dw4();
    // pw4();
    // conv2d_5();
    // dw5();
    // pw5();
    // add_3();
    // conv2d_6();
    // dw6();
    // pw6();
    // add_4();
    // conv2d_7();
    // dw7();
    // pw7();
    // //Added start
    // conv2d_8();
    // dw8();
    // pw8();
    // //Added stop
    // add_5();
    // conv2d_9();
    // dw9();
    // pw9();
    // add_6();
    // conv2d_10();
    // dw10();
    // pw10();
    // add_7();
    // conv2d_11();
    // dw11();
    // pw11();
    // conv2d_12();
    // dw12();
    // pw12();
    // add_8();
    // conv2d_13();
    // dw13();
    // pw13();
    // conv2d_14();
    // dw14();
    // pw14();
    // add_9();
    // conv2d_15();
    // dw15();
    // pw15();
    // add_10();
    // conv2d_16();
    // averagepool2d_1();
    // conv2d_17();
    // averagepool2d_2();
    // conv2d_18();
    // reshape();
    // softmax();
    double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Done in %f seconds\n", elapsed_time);
    return 0;
}
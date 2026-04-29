#include "mnist_accelerator.h"
#include "weights.h"

void mnist_accelerator(data_t input_img[784], int &prediction) {
    
   // #pragma HLS allocation amount=64 instances=mul limit=operation
    
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
    #pragma HLS INTERFACE m_axi port=input_img depth=784 bundle=gmem offset=slave
    #pragma HLS INTERFACE s_axilite port=prediction bundle=CTRL_BUS

    // Local buffers
    data_t local_img[784];
    data_t pool1_out[4][14][14];
    data_t conv2_out[8][14][14];

    // Array Partitioning: Factor 8 matches the 8 channels of CONV2
    // This allows the FC layer to read all channels for a single (i,j) coordinate at once
    #pragma HLS ARRAY_PARTITION variable=conv2_out cyclic factor=8 dim=1
    
    #pragma HLS BIND_STORAGE variable=local_img type=RAM_1P impl=bram
    #pragma HLS BIND_STORAGE variable=pool1_out type=RAM_1P impl=bram
    #pragma HLS BIND_STORAGE variable=conv2_out type=RAM_1P impl=bram

    typedef ap_fixed<24, 12> acc_t; 
    const acc_t SCALE = 0.00390625; 

    // 1. Burst Load
    LOAD_INPUT: for(int i=0; i<784; i++) {
        #pragma HLS PIPELINE II=1
        local_img[i] = input_img[i];
    }

    // 2. CONV1 + POOL (Already well-optimized for sharing)
    CONV1_F: for (int f = 0; f < 4; f++) {
        CONV1_R: for (int i = 0; i < 28; i += 2) {
            CONV1_C: for (int j = 0; j < 28; j += 2) {
                #pragma HLS PIPELINE II=4 
                acc_t sum = (acc_t)conv1_bias[f] * SCALE;
                for (int r = -1; r <= 1; r++) {
                    for (int c = -1; c <= 1; c++) {
                        int row = i + r;
                        int col = j + c;
                        if (row >= 0 && row < 28 && col >= 0 && col < 28) {
                            acc_t w = (acc_t)conv1_weight[(f * 9) + ((r + 1) * 3) + (c + 1)] * SCALE;
                            sum += (acc_t)local_img[row * 28 + col] * w;
                        }
                    }
                }
                pool1_out[f][i/2][j/2] = (sum > (acc_t)0) ? (data_t)sum : (data_t)0;
            }
        }
    }

    // 3. CONV2
    CONV2_F: for (int f = 0; f < 8; f++) {
        CONV2_R: for (int i = 0; i < 14; i++) {
            CONV2_C: for (int j = 0; j < 14; j++) {
                #pragma HLS PIPELINE II=8
                acc_t sum = (acc_t)conv2_bias[f] * SCALE;
                for (int c_in = 0; c_in < 4; c_in++) {
                    for (int r = -1; r <= 1; r++) {
                        for (int c = -1; c <= 1; c++) {
                            int row = i + r;
                            int col = j + c;
                            if (row >= 0 && row < 14 && col >= 0 && col < 14) {
                                acc_t w = (acc_t)conv2_weight[(f * 36) + (c_in * 9) + ((r + 1) * 3) + (c + 1)] * SCALE;
                                sum += (acc_t)pool1_out[c_in][row][col] * w;
                            }
                        }
                    }
                }
                conv2_out[f][i][j] = (sum > (acc_t)0) ? (data_t)sum : (data_t)0;
            }
        }
    }

    // 4. FC Layer - Area Optimized
    acc_t max_score = -2048; 
    int best_digit = 0;

    FC_OUT: for (int out = 0; out < 10; out++) {
        acc_t sum = (acc_t)fc_bias[out] * SCALE;
        
        // Flattened inner loops to prevent automatic unrolling of the 1568 elements
        FC_ACCUM: for (int k = 0; k < 1568; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL off
            
            // Extract indices from flat k
            int f = k / 196;
            int rem = k % 196;
            int i = rem / 14;
            int j = rem % 14;

            acc_t w = (acc_t)fc_weight[(out * 1568) + k] * SCALE;
            sum += (acc_t)conv2_out[f][i][j] * w;
        }

        if (sum > max_score) {
            max_score = sum;
            best_digit = out;
        }
    }
    prediction = best_digit;
}
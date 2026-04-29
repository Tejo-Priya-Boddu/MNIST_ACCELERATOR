#include "mnist_accelerator.h"
#include "weights.h" 

// 1. Downsampler 
void downsampler(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            #pragma HLS PIPELINE II=1
            data_t val = in_stream.read();
            if (i % 2 == 0 && j % 2 == 0) {
                out_stream.write(val);
            }
        }
    }
}

// 2. Conv Engine 
void conv_engine(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream) {
    for (int i = 0; i < 196; i++) { 
        #pragma HLS PIPELINE II=1
        data_t pixel = in_stream.read();
        
        
        data_t weight = (data_t)conv1_weight[i % 36]; 
        data_t result = (pixel * weight) + (data_t)conv1_bias[0]; 
        
        out_stream.write(result);
    }
}

// 3. FC Classifier 
void fc_classifier(hls::stream<data_t>& in_stream, int &prediction) {
    acc_t scores[10] = {0};
    #pragma HLS ARRAY_PARTITION variable=scores complete 
    
    for (int i = 0; i < 196; i++) {
        #pragma HLS PIPELINE II=1
        data_t val = in_stream.read();
        
        for(int d = 0; d < 10; d++) {
          
            scores[d] += val * (data_t)fc_weight[i * 10 + d]; 
        }
    }
    
    // Argmax logic
    acc_t max_v = -32768; 
    int best = 0;
    for(int d = 0; d < 10; d++) {
        if(scores[d] > max_v) {
            max_v = scores[d];
            best = d;
        }
    }
    prediction = best;
}
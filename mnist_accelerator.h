#ifndef MNIST_ACCEL_H
#define MNIST_ACCEL_H

#include <ap_fixed.h>
#include <hls_stream.h>


typedef ap_fixed<16, 8> data_t; 
typedef ap_fixed<24, 12> acc_t;

// Top level function declaration
void mnist_accelerator(data_t* input_img, int &prediction);

// Modular function prototypes
void stream_loader(data_t* in, hls::stream<data_t>& out_stream);
void downsampler(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream);
void conv_engine(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream);
void fc_classifier(hls::stream<data_t>& in_stream, int &prediction);

#endif
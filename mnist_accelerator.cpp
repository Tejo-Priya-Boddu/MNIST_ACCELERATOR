#include "mnist_accelerator.h"
#include "weights.h"

// EXTENSION: Task-Level Parallelism via Dataflow (Surpasses Oraon 2024 sequential flow)
void mnist_accelerator(data_t input_img[784], int &prediction) {
    #pragma HLS INTERFACE m_axi port=input_img depth=784 bundle=gmem offset=slave
    #pragma HLS INTERFACE s_axilite port=prediction bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

    // Internal FIFOs for Streaming
    hls::stream<data_t> load_stream("load_stream");
    hls::stream<data_t> down_stream("down_stream");
    hls::stream<data_t> conv_stream("conv_stream");

// EXTENSION: Set FIFO depths to prevent deadlocks and optimize throughput
    // 784 is a full 28x28 image buffer; 196 is a 14x14 buffer
    #pragma HLS STREAM variable=load_stream depth=784
    #pragma HLS STREAM variable=down_stream depth=196
    #pragma HLS STREAM variable=conv_stream depth=196

    // EXTENSION: Strategic hardware folding limit
    #pragma HLS allocation amount=64 instances=mul limit=operation

    #pragma HLS DATAFLOW
    
    // 1. Burst Read from Memory
    stream_loader(input_img, load_stream);

    // 2. EXTENSION: Downsampler (Implements Kwon & Kim 14x14 logic)
    downsampler(load_stream, down_stream);

    // 3. Streaming Convolution
    conv_engine(down_stream, conv_stream);

    // 4. Final Classification
    fc_classifier(conv_stream, prediction);
}

// Loader implementation
void stream_loader(data_t* in, hls::stream<data_t>& out_stream) {
    for(int i = 0; i < 784; i++) {
        #pragma HLS PIPELINE II=1
        out_stream.write(in[i]);
    }
}
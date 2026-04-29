#ifndef MNIST_ACCEL_H
#define MNIST_ACCEL_H

#include <ap_fixed.h>

// Use 16-bit fixed point as defined by your trainee
typedef ap_fixed<24, 12> data_t;
// typedef ap_fixed<32,16> acc_t;

// Change from hls::stream to data_t array
void mnist_accelerator(data_t input_img[784], int &prediction);

#endif
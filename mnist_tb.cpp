#include <iostream>
#include "mnist_accelerator.h"


void generate_mock_digit(data_t input[784]) {
    for(int i=0; i<784; i++) input[i] = 0;
    
    for(int j=5; j<23; j++) input[10*28 + j] = 0.9;
    
    for(int i=10; i<25; i++) input[i*28 + (28-i)] = 0.9;
}

int main() {
    std::cout << "--- STARTING MNIST ULTRA96-V2 EXTENSION TESTBENCH ---" << std::endl;

    data_t test_image[784];
    int hw_prediction = -1;
    
    // 1. Setup Input Data
    generate_mock_digit(test_image);

    // 2. Call the Hardware IP (The Top Function)
    
    mnist_accelerator(test_image, hw_prediction);

    // 3. Extension Verification Logic
    std::cout << "Hardware Prediction: " << hw_prediction << std::endl;

    // We check if the result is valid (0-9)
    if (hw_prediction >= 0 && hw_prediction <= 9) {
        std::cout << "TEST PASSED: Valid classification detected." << std::endl;
        
        return 0;
    } else {
        std::cout << "TEST FAILED: Invalid prediction." << std::endl;
        return 1;
    }
}
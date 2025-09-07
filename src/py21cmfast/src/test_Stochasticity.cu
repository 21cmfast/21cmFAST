#include <cassert>
#include <iostream>

#include "Stochasticity.cu"

void testCondenseDeviceArray()
{
    // Input data
    float h_array[] = {1.0f, 0.0f, 2.0f, 3.0f, 0.0f, 4.0f};
    float mask_value = 0.0f;
    int original_size = 6;

    // Expected outputs
    float expected_array[] = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f};
    int expected_valid_size = 4;

    // Allocate and copy to device
    float *d_array;
    cudaMalloc(&d_array, original_size * sizeof(float));
    cudaMemcpy(d_array, h_array, original_size * sizeof(float), cudaMemcpyHostToDevice);

    // Call the function from Stochasticity.cu
    int valid_size = condenseDeviceArray(d_array, original_size, mask_value);

    // Copy the results back to the host
    float h_result[original_size];
    cudaMemcpy(h_result, d_array, original_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate the results
    assert(valid_size == expected_valid_size);
    for (int i = 0; i < original_size; ++i)
    {
        assert(h_result[i] == expected_array[i]);
    }

    std::cout << "Test passed: condenseDeviceArray\n";

    // Free device memory
    cudaFree(d_array);
}

int main()
{
    testCondenseDeviceArray();
    return 0;
}

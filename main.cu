
#include <cstdio>
#include <cuda_runtime.h>


__global__ void findMinWithRaceConditions(const int* d_arr, int n, int* d_min) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // Global thread ID
    if (tid >= n) return;

    // Threads race to update the global minimum
    int localValue = d_arr[tid];
    if (localValue < *d_min) {
        *d_min = localValue; // Potential race condition
    }
}

// Host function to stabilize the result
void stabilizeGlobalMin(const int* h_arr, int n, int& h_min) {
    int* d_arr;
    int* d_min;

    // Allocate memory on the device
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_min, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel multiple times to stabilize the result
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    for (int iter = 0; iter < 10; ++iter) { // Fixpoint computation loop
        findMinWithRaceConditions<<<gridSize, blockSize>>>(d_arr, n, d_min);
        cudaDeviceSynchronize(); // Synchronize between kernel launches
    }

    // Copy result back to host
    cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_min);
}

// Sequential Version
int find_array_min(int array[], int size){
    int min = array[0];
    for (int i = 1; i < size; i++){
        if (array[i] < min){
            min = array[i];
        }
    }
    return min;
}

int main() {
     int h_min = INT_MAX;


	const int n = 1'000'000; // Large array to increase potential for race conditions
    int* h_arr = new int[n];
    int verified_min = INT_MAX; // Variable used to see if the algorithms work

    // Initialize the array with random values while tracking the minimum
    for (int i = 0; i < n; ++i) {
        h_arr[i] = std::rand() % 100000 - 50000; // Random values between -50000 and 49999
        if (h_arr[i] < verified_min) {
            verified_min = h_arr[i]; // Update the verified minimum during initialization
        }
    }


    int min = find_array_min(h_arr, n);

    printf("Min value in array is: %d\n", min);

	stabilizeGlobalMin(h_arr, n, h_min);

  	std::cout << "Minimum value (Race Conditions Allowed): " << h_min << std::endl;

    return 0;
};

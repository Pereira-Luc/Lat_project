#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>
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

__global__ void findMinCUDA(const int* d_arr, int n, int* d_min) {
    extern __shared__ int sharedMin[]; 

    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    int localTid = threadIdx.x;                    

    // Initialize shared memory
    if (tid < n)
        sharedMin[localTid] = d_arr[tid];
    else
        sharedMin[localTid] = INT_MAX; // For threads beyond array bounds
    __syncthreads();

    // Perform block-wide reduction to find minimum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localTid < stride && tid + stride < n) {
            sharedMin[localTid] = min(sharedMin[localTid], sharedMin[localTid + stride]);
        }
        __syncthreads();
    }

    // First thread of each block updates the global minimum
    if (localTid == 0) {
        atomicMin(d_min, sharedMin[0]);
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
    const int blockSize = 1024;
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
int find_array_min(int array[], int size) {
    printf("Starting sequential minimum search.\n");
    int min = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] < min) {
            min = array[i];
        }
    }
    return min;
}

int main() {
    printf("Starting program.\n");

    int h_min = INT_MAX;

    const int n = 1'000'000; // Large array to increase potential for race conditions
    int* h_arr = new int[n];
    int verified_min = INT_MAX; // Variable used to see if the algorithms work

    printf("Initializing array with random values.\n");
    // Initialize the array with random values while tracking the minimum
    for (int i = 0; i < n; ++i) {
        h_arr[i] = std::rand() % 100000 - 50000; // Random values between -50000 and 49999
        if (h_arr[i] < verified_min) {
            verified_min = h_arr[i]; // Update the verified minimum during initialization
        }
    }

    printf("Starting sequential minimum computation.\n");
    int min = find_array_min(h_arr, n);
    printf("Sequential minimum value: %d\n", min);

    printf("Starting CUDA minimum computation with race conditions.\n");
    stabilizeGlobalMin(h_arr, n, h_min);

    printf("Minimum value (Race Conditions Allowed): %d\n", h_min);

    delete[] h_arr;
    printf("Freed host memory for array.\n");

    return 0;
}

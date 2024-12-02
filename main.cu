
#include <cstdio>
#include <cuda_runtime.h>


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
    int array[] = {1, 2, 3, 4, 5, -7};
    int size = sizeof(array) / sizeof(array[0]);

    int min = find_array_min(array, size);

    printf("Min value in array is: %d\n", min);

    return 0;
};

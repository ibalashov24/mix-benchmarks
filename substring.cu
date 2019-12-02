#include <stdio.h>
#include <string.h>

#define BLOCKS  16
#define THREADS 64

__device__ int isSubstring = 0;

__global__ 
void findSubstring(
		const char *string, 
		const char *sample, 
		int stringLength, 
		int patternLength)
{
	// Number of symbols of the string processed in this thread
    int packSize = stringLength > BLOCKS*THREADS ? stringLength / (BLOCKS*THREADS) : 1;
	// Index of the beginning of the current block of processed symobols
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * packSize;

    if (index + packSize > BLOCKS*THREADS)
    {
        packSize = BLOCKS*THREADS - index;
    }
	// If no more left to process
    if (packSize == 0)
    {
        return;
    }

    for (int i = index; i < index + packSize; ++i)
    {   
        if (i + patternLength >= stringLength)
        {
            break;
        }

        int j;
        for (j = 0; j < patternLength; ++j)
        {
            if (string[i + j] != sample[j])
            {
                break;
            }
        } 

        if (j == patternLength)
        {
            isSubstring = 1;
        }
    }
}


int main()
{
    char *string, *pattern;
    cudaMalloc((void **) &string, 11);
    cudaMalloc((void **) &pattern, 5);

    char *tempString, *tempPattern;
    tempString = (char *) malloc(11);
    tempPattern = (char *) malloc(5);
    strcpy(tempString, "abcdefghij");
    strcpy(tempPattern, "defg");

    cudaMemcpy((void *) string, (void *) tempString, 11, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) pattern, (void *) tempPattern, 5, cudaMemcpyHostToDevice);

    findSubstring<<<BLOCKS, THREADS>>>(string, pattern, 10, 4);

	cudaDeviceSynchronize();

	int *result;
	result = (int *) malloc(sizeof(int));
	cudaMemcpyFromSymbol((void *) result, isSubstring, sizeof(int), 0, cudaMemcpyDeviceToHost);
    printf("%d\n", *result);

    return 0;
}

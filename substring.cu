#include <stdio.h>

#define BLOCKS  16
#define THREADS 64

__global__ 
void findSubstring(
		const char *string, 
		const char *sample, 
		int stringLength, 
		int patternLength,
		volatile int *result)
{
    int packSize =  
        stringLength < BLOCKS*THREADS ? stringLength / (BLOCKS*THREADS) : 1;
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * packSize;

    if (index + packSize > BLOCKS*THREADS)
    {
        packSize = BLOCKS*THREADS - index;
    }
    if (packSize == 0)
    {
        return;
    }

    //isSubstring = 10;

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
//            isSubstring = 1;
        }
    }
}


int main()
{
	int *result;
    cudaMalloc((void **) &result, sizeof(int));
	cudaMemset((void *) result, 0, sizeof(int));
	
/*    char *string, *pattern;
    cudaMalloc((void **) &string, 11);
    cudaMalloc((void **) &pattern, 5);

    cudaMemcpy((void **) &string, (const void *) "abcdefghij\0", 11, cudaMemcpyHostToDevice);
    cudaMemcpy((void **) &pattern, (const void *) "defg\0", 5, cudaMemcpyHostToDevice);

    findSubstring<<<BLOCKS, THREADS>>>(string, pattern, 10, 4);*/

    int *kek;
	kek = (int *) malloc(sizeof(int));
	*kek = 5;
	cudaMemcpy((void *) kek, (void *) result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", *kek);

    return 0;
}

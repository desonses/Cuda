/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/



#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>

/*
muestra los hilos disponibles por grid en tu targeta cuda
*/

// estructura de dim3
// dim3 blocks(Bx, By, Bz);
// dim3 threads(hx, hy, hz);

gridDim.x = Bx
gridDim.y = By
gridDim.z = Bz
blockDim.x = hx
blockDim.y = hy
blockDim.z = hz



int main(int argc, char** argv)
{
	cudaDeviceProp deviceProp;
	int deviceID;
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&deviceProp, deviceID);
	printf("MAX threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("MAX BLOCK SIZE\n");
	printf(" [x -> %d]\n [y -> %d]\n [z -> %d]\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("MAX GRID SIZE\n");
	printf(" [x -> %d]\n [y -> %d]\n [z -> %d]\n"), deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2];
	

	//dim3 bloques(3, 2, 1);
	//dim3 hilos(16, 16, 1);
	
	printf("\n pulsa INTRO parsa finalizar...");
	fflush(stdin);
	char tecla = getchar();


	return 0;


}


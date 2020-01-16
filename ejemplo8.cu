/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/



#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <cuda.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 33

/*
realiza la transpuesta de una matriz
*/

// definicio de memoria constante CUDA
__constant__ float dev_A[N][N];

//GLOBAL: func desde el host y ejecutada en el kernel(DEVICE)

__global__ void transpuesta(float *dev_B) 
{
	int columna = threadIdx.x;
	int fila = threadIdx.y;
	int pos = columna + N * fila;
	
	// cada hilo coloca un elemento de la matriz final
	dev_B[pos] = dev_A[columna][fila];
}


int main(int argc, char** argv)
{
	float *hst_A, *hst_B;
	float *dev_B;
	int size = N * N * sizeof(float);

	//reserva de memoria en el host
	hst_A = (float*)malloc(size);
	hst_B = (float*)malloc(size);

	//reserva de memoria en el device
	cudaMalloc((void**)&dev_B, size);

	//llenar la matriz

	for (int i = 0; i < N*N; i++)
	{
		hst_A[i] = float(i) + 1;

	}

	//copiar los datos hacia el device
	cudaError_t error = cudaMemcpyToSymbol(dev_A, hst_A, size);
	if (error != cudaSuccess) {
		printf("Error Memori const\n");
	}

	//dimensiones del kernel a lanzar
	dim3 bloques(1);
	dim3 hilos(N, N);

	//lanzamiento del kernel
	transpuesta <<<bloques, hilos >>> (dev_B);

	//recoger los datos
	cudaMemcpy(hst_B, dev_B, size, cudaMemcpyDeviceToHost);

	//impresion de los datos
	printf("Matriz original:\n");
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			printf("%.2f\t", hst_A[j + i * N]);
		}
		printf("\n");
	}
	
	printf("Matriz transpuesta:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%.2f\t", hst_B[j + i * N]);
		}
		printf("\n");
	}
	
	//
	printf("\n pulsa INTRO para salir:\n");
	fflush(stdin);
	char tecla = getchar();

	return 0;
}

/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/

#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 3
/*
multiplicacion de matrices con memoria constante
*/

// definicion de memoria constante CUDA

__constant__ float dev_A[N][N];
__constant__ float dev_B[N][N];


//GLOBAL: func desde el host y ejecutada en el kernel(DEVICE)


__global__ void multiplicacion(float *dev_C)
{
	int suma = 0;
	int columna = threadIdx.x;
	int fila = threadIdx.y;
	int pos = columna + N * fila;

	if (columna < N && fila < N) 
	{
		for (int k = 0; k < N; k++) 
		{
			dev_C[pos] += dev_A[fila][k] * dev_B[k][columna];
		}
	}
}



int main(int argc, char** argv)
{
	float *hst_A, *hst_B, *hst_C;
	float *dev_C;
	int size = N * N * sizeof(float);

	//reserva de memoria en el host
	hst_A = (float*)malloc(size);
	hst_B = (float*)malloc(size);
	hst_C = (float*)malloc(size);
	//reserva de memoria en el device
	cudaMalloc((void**)&dev_C, size);

	//llenar la matriz
	for (int i = 0; i < N*N; i++)
	{
		hst_A[i] = float(i) + 1;
		hst_B[i] = float(i);
	}

	//copiar los datos hacia el device desde memoria constante
	cudaError_t error = cudaMemcpyToSymbol(dev_A, hst_A, size);
	if (error != cudaSuccess) {
		printf("Error Memoria constante dev_A to hst_A\n");
	}
	error = cudaMemcpyToSymbol(dev_B, hst_B, size);
	if (error != cudaSuccess) {
		printf("Error Memoria constante dev_B to hst_B\n");
	}

	//dimensiones del kernel a lanzar
	dim3 bloques(1);
	dim3 hilos(N, N);

	//lanzamiento del kernel
	multiplicacion <<<bloques, hilos >>> (dev_C);

	//recoger los datos
	cudaMemcpy(hst_C, dev_C, size, cudaMemcpyDeviceToHost);

	//impresion de los datos
	printf("\nMatriz A:\n");

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			printf("%.2f\t", hst_A[j + i * N]);
		}
		printf("\n");
	}
	printf("\nMatriz B:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			printf("%.2f\t", hst_B[j + i * N]);
		}
		printf("\n");
	}
	printf("\n");
	printf("multiplicacion de matrices A y B:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%.2f\t", hst_C[j + i * N]);
		}
		printf("\n");
	}
	
	printf("\n pulsa INTRO para salir:\n");
	fflush(stdin);
	char tecla = getchar();

	return 0;
}


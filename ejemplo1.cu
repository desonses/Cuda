/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_fp16.h"


/*
En el siguiente ejemplo se muestran las diferencias y las similitudes que existen a
la hora de reservar memoria tanto en el host como en el device. En este ejemplo se
reserva espacio para una matriz cuadrada de NxN elementos, se inicializa
en el host con valores aleatorios (entre 0 y 9) de tipo float y despues se transfieren los datos
desde el host hasta el device:
*/


#define N 8
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// declaracion
	float *hstA_matriz;
	float *devA_matriz;
	float *hstB_matriz;////
	float *devB_matriz;////

	// reserva en el host
	hstA_matriz = (float*)malloc(N*N * sizeof(float));
	hstB_matriz = (float*)malloc(N*N * sizeof(float));/////
	// reserva en el device
	cudaMalloc((void**)&devA_matriz, N*N * sizeof(float));
	cudaMalloc((void**)&devB_matriz, N*N * sizeof(float));////
	// inicializacion de datos
	srand((int)time(NULL));
	for (int i = 0; i < N*N; i++)
	{
		hstA_matriz[i] = (float)(rand() % 2);
	}

	// copia de datos
	cudaMemcpy(devA_matriz, hstA_matriz, N*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devB_matriz, devA_matriz, N*N * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(hstB_matriz, devB_matriz, N*N * sizeof(float), cudaMemcpyDeviceToHost);


	// salida
	printf("matriz A\n");
	for (int j = 0; j < N*N; j++)
	{

		printf("%f, ",hstA_matriz[j]);
	}
	printf("\n\n");
	printf("matriz B\n");
	for (int k = 0; k < N*N; k++)
	{

		printf("%f, ", hstB_matriz[k]);
	}

	cudaFree(devA_matriz);
	cudaFree(devB_matriz);
	printf("\npulsa INTRO para finalizar...");
	fflush(stdin);
	char tecla = getchar();
	return 0;
}



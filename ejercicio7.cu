/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/


#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads();
#else
#define cuda_SYNCTHREADS()
#endif

#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <cuda.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
sincronizacion de hilos, verificacion de posibles errores, 
suma las potencias de elementos de un vector en el orden de log2(n)
*/


#define N 8
__device__ float valores(float, float);

__host__ void check_CUDA_Error(const char *mensaje)
{
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
	}
}


__global__ void reduccion(float *vector, float *suma)
{
	//reserva de memoria en la zona de memoria compartida
	__shared__ float temporal[N];

	//indice local de cada hilo -> kernel con un solo bloque
	int Id = threadIdx.x;

	//copiamos en 'temporal' el vector y sincronizamos los hilos
	temporal[Id] = vector[Id];
	
	cuda_SYNCTHREADS();
	//reduccion paralela
	int salto = N / 2;

	//realizamos log2(N) iteraciones
	while (salto)
	{
		//solo trabajan la mitad de los hilos
		if (Id < salto)
		{	

			temporal[Id] = (1 / powf(temporal[Id], 2)) + (1 / powf(temporal[Id + salto], 2));
			printf("temporal: %.3f\n", temporal[Id]);
		}
		//cuda_SYNCTHREADS();
		cuda_SYNCTHREADS();
		salto = salto / 2;
	}
	//el hilo 0 escribe el resultado final en la  memoria global
	if (Id == 0)
	{
		*suma = temporal[Id];
	}
}

__device__ float valores(float valor1, float valor2) {

	float suma = (1 / pow(valor1, 2)) + (1 / pow(valor2, 2));
	return suma;
}

int main(int argc, char** argv) 
{
	float *vector1, *resultado;
	float *dev_vector1, *dev_resultado;
	int size = N * sizeof(float);
	//reserva de memoria en el host
	vector1 = (float*)malloc(size);
	resultado = (float*)malloc(size);

	//reserva de memoria en el device
	cudaMalloc((void**)&dev_vector1, size);
	cudaMalloc((void**)&dev_resultado, size);

	// inicializacion de los vectores
	for (int i = 0; i < N; i++) {
		vector1[i] = (float)i + 1;
	}

	//enviar los datos hacia el Device
	cudaMemcpy(dev_vector1, vector1, size, cudaMemcpyHostToDevice);

	//lanzamiento del kernel

	reduccion<<<1, N>>>(dev_vector1, dev_resultado);

	//recogida de los datos

	cudaMemcpy(resultado, dev_resultado, size, cudaMemcpyDeviceToHost);

	//impresion de los datos
	printf("\n>vector1: \n");
	for (int i = 0; i < N; i++) {
		printf("%.3f, ", 1/pow(vector1[i],2));

	}

	printf("\n");
	printf(">suma: \n");
	for (int i = 0; i < N; i++) {
		printf("%.3f, ", resultado[i]);

	}

	printf("\n");

	//liberacion de memoria del device y host

	cudaFree(dev_vector1);
	cudaFree(dev_resultado);
	free(vector1);
	free(resultado);
	printf("\n...");
	fflush(stdin);
	char tecla = getchar();

	return 0;

}




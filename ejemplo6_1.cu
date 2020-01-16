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

#define N 20

/*
operaciones con matrices
*/

//GLOABL: funcion llamada desde el host y ejecutada en el device (kernel)

__global__ void suma(float *A, float *B, float *C)
{
	//indice de las columnas
	int columna = threadIdx.x;
	//indice de las filas
	int fila = threadIdx.y;
	//indice lineal
	int Id = columna + fila * blockDim.x;
	//sumamos cada elemento
	C[Id] = A[Id] + B[Id];
}


//Cada elemento de la matriz C se obtiene como la suma de los
//elementos de la matriz A ubicados en posiciones adyacentes

__global__ void add(float *A, float *C) 
{
	int columna = threadIdx.x;
	//indice de las filas
	int fila = threadIdx.y;
	//indice lineal
	int Id = columna + fila * blockDim.x;

	int id1 = (columna - 1) + fila * blockDim.x;
	int id2 = (columna + 1) + fila * blockDim.x;
	int id3 = columna + (fila - 1) * blockDim.x;
	int id4 = columna + (fila + 1) * blockDim.x;

	if ((fila > 0 && fila < N - 1) && (columna > 0 && columna < N - 1)) {

		C[Id] = A[id1] + A[id2] + A[id3] + A[id4];
	}
	else
	{
		C[Id] = A[Id];
	}
}


// funcion chequeo de errores

__host__ void check_CUDA_Error(const char *mensaje) 
{
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
	}
}



int main(int argc, char** argv) 
{
	//declaracions
	float *hst_A, *hst_B, *hst_C;
	float *dev_A, *dev_B, *dev_C;
	int size = N * N * sizeof(float);

	//reserva de memoria en el host
	hst_A = (float*)malloc(size);
	hst_B = (float*)malloc(size);
	hst_C = (float*)malloc(size);

	//reserva de memoria en el device
	cudaMalloc((void**)&dev_A, size);
	check_CUDA_Error("Error malloc dev_A!");
	cudaMalloc((void**)&dev_B, size);
	check_CUDA_Error("Error malloc dev_B!");
	cudaMalloc((void**)&dev_C, size);
	check_CUDA_Error("Error malloc dev_C!");

	//inicializacion de los vectores

	for (int i = 0; i < N*N; i++) {

		hst_A[i] = (float)(rand() % 5);
		hst_B[i] = (float)(rand() % 5);
	}

	//enviar datos del hosto al device
	
	cudaMemcpy(dev_A, hst_A, size, cudaMemcpyHostToDevice);
	check_CUDA_Error("Error Memcpy hst_A To dev_A");
	cudaMemcpy(dev_B, hst_B, size, cudaMemcpyHostToDevice);
	check_CUDA_Error("Error Memcpy hst_B To dev_B");
	//cudaMemcpy(dev_C, hst_C, N*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_C, hst_C, size, cudaMemcpyHostToDevice);
	check_CUDA_Error("Error Memcpy hst_C To dev_C");
	//dimenciones del kernel
	dim3 Nbloques(1);
	dim3 hilosB(N, N);

	//////MEDICION DE TIEMPO EN GPU///////////////
	// declaracion de eventos para medir el tiempo de ejecucion en la GPU
	cudaEvent_t start;
	cudaEvent_t stop;
	// creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// marca de inicio
	cudaEventRecord(start, 0);
	//llamada al kernel dibimensional de NxN hilos
	//suma <<<Nbloques, hilosB >>> (dev_A, dev_B, dev_C);
	add<<<Nbloques, hilosB >>>(dev_A, dev_C);
	check_CUDA_Error("Error kernel");
	
	// marca de final
	cudaEventRecord(stop, 0);
	// sincronizacion GPU-CPU
	cudaEventSynchronize(stop);
	// calculo del tiempo en milisegundos
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	////////MEDICION DE TIEMPO EN GPU/////////////////

	//recodiga de los datos
	cudaMemcpy(hst_C, dev_C, size, cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error Memcpy dev_C To hst_C");
	//impresion del resultado
	
	printf("A:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			printf("%.2f\t", hst_A[j + i * N]);
		}
		printf("\n");
	}

	
	printf("B:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			printf("%.2f\t", hst_B[j + i * N]);
		}
		printf("\n");
	}
	
	printf("C:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			printf("%.2f\t", hst_C[j + i * N]);
		}
		printf("\n");
	}
	printf("\n\n");
	printf("> Tiempo de ejecucion: %f ms\n", elapsedTime);

	cudaFree(hst_A);
	cudaFree(hst_B);
	cudaFree(hst_C);
	free(hst_A);
	free(hst_B);
	free(hst_C);

	printf("\n pulsa INTRO parsa finalizar...");
	fflush(stdin);
	char tecla = getchar();

	return 0;
}


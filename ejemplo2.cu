/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define N 16 //tamano de los vectores
#define BLOCKS 5 // tamano del bloque(numero de hilos en cada bloque)


gridDim.x: La primera nos da el n�mero de bloques (M)
blockDim.x: la segunda el n�mero de hilos que tiene cada bloque (N)

/*
En este este ejemplo se realiza un ejemplo sencillo de suma de vectores (entrada x entrada)
*/

//Global: funcion llamada desde el host y ejecutada en el device(kernel)
__global__ void Add(float *a, float *b, float *c)
{
	int Id = threadIdx.x + blockDim.x * blockIdx.x;
	printf("(%d, %d, %d) ", threadIdx.x, blockDim.x, blockIdx.x);
	printf("hilo: %d, ", Id);
	//solo trabajan los N hilos
	if (Id < N) {
		c[Id] = a[Id] * b[Id];
	}
}



int main(int argc, char **argv)
{
	float *vector1, *vector2, *resultado;
	float *dev_vector1, *dev_vector2, *dev_resultado;

	//reserva de memoria en el host
	vector1 = (float*)malloc(N * sizeof(float));
	vector2 = (float*)malloc(N * sizeof(float));
	resultado = (float*)malloc(N * sizeof(float));

	//reserva de memoria en el device
	cudaMalloc((void**)&dev_vector1, N * sizeof(float));
	cudaMalloc((void**)&dev_vector2, N * sizeof(float));
	cudaMalloc((void**)&dev_resultado, N * sizeof(float));

	// inicializacion de los vectores
	for (int i = 0; i < N; i++) {
		vector1[i] = (float) rand() / RAND_MAX;
		vector2[i] = (float) rand() / RAND_MAX;
	}

	//enviar los datos hacia el Device
	cudaMemcpy(dev_vector1, vector1, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vector2, vector2, N * sizeof(float), cudaMemcpyHostToDevice);

	//lanzamiento del kernel
	//calculamos el numero de bloques necesario para un tamano de bloque fijo 
	int nBloques = N / BLOCKS;
	if (N % BLOCKS != 0) {
		nBloques = nBloques + 1;

	}


	int hilosB = BLOCKS;
	printf("\n");
	printf("vector de %d elementos\n", N);
	printf("Lanzamiento con %d bloques (%d hilos)\n", nBloques, nBloques*hilosB);

	Add <<<nBloques,hilosB>>>(dev_vector1, dev_vector2, dev_resultado);

	//recogida de los datos

	cudaMemcpy(resultado, dev_resultado, N*sizeof(float),cudaMemcpyDeviceToHost);

	//impresion de los datos
	printf(">vector1: \n");
	for (int i = 0; i < N;i++) {
		printf("%.2f, ", vector1[i]);

	}
	printf("\n");
	printf(">vector2: \n");
	for (int i = 0; i < N; i++) {
		printf("%.2f, ", vector2[i]);

	}
	printf("\n");
	printf(">suma: \n");
	for (int i = 0; i < N; i++) {
		printf("%.2f, ", resultado[i]);

	}

	printf("\n");

	//liberacion de memoria del device y host

	cudaFree(dev_vector1);
	cudaFree(dev_vector2);
	cudaFree(dev_resultado);
	free(vector1);
	free(vector2);
	free(resultado);
	printf("\n...");
	fflush(stdin);
	char tecla = getchar();

	return 0;

}




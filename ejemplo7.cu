/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/


#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif

#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define N 16
/*
suma de los elementos de un vector en el orden de log2(n)
*/

//kernel

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
			temporal[Id] = temporal[Id] + temporal[Id + salto];
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


int main(int argc, char** argv)
{

	float *vector1,  *resultado;
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
		vector1[i] = (float)rand() / RAND_MAX;

	}

	//enviar los datos hacia el Device
	cudaMemcpy(dev_vector1, vector1, size, cudaMemcpyHostToDevice);

	//lanzamiento del kernel
	
	reduccion <<<1, N >>> (dev_vector1, dev_resultado);

	//recogida de los datos

	cudaMemcpy(resultado, dev_resultado, size, cudaMemcpyDeviceToHost);

	//impresion de los datos
	printf(">vector1: \n");
	for (int i = 0; i < N; i++) {
		printf("%.3f, ", vector1[i]);

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

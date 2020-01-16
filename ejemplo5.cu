/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/



#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
ejemplo que llena un vector que invierte sus valores
*/

#define N 30 //tamano de los vectores

__global__ void invierte(float *a, float *b) {
	int id = threadIdx.x;
	//int id = threadIdx.x + blockDim.x * blockIdx.x;// para n-bloques de 1 hilo

	if (id < N) 
	{
		b[id] = a[N-id];
	}
}



__host__ void check_CUDA_Error(const char *mensaje) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
	}
}


int main(int argc, char** argv)
{	
	float *vector1, *resultado;
	float *dev_vector1, *dev_resultado;

	//reserva de memoria en el host
	vector1 = (float*)malloc(N * sizeof(float));
	resultado = (float*)malloc(N * sizeof(float));

	//reserva de memoria en el device
	cudaMalloc((void**)&dev_vector1, N * sizeof(float));
	check_CUDA_Error("Error Malloc dev_vector");
	cudaMalloc((void**)&dev_resultado, N * sizeof(float));
	check_CUDA_Error("Error Malloc dev_resultado");

	// inicializacion de los vectores
	printf("vector inicial: \n");
	for (int i = 0; i < N; i++) {
		vector1[i] = (float)rand() / RAND_MAX;
		printf("%.2f, ", vector1[i]);
	}
	
	//enviar los datos hacia el Device
	cudaMemcpy(dev_vector1, vector1, N * sizeof(float), cudaMemcpyHostToDevice);
	check_CUDA_Error("Error CudaMemcpy");
	
	//MEDICION DE TIEMPO EN GPU
	// declaracion de eventos para medir el tiempo de ejecucion en la GPU
	cudaEvent_t start;
	cudaEvent_t stop;
	// creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// marca de inicio
	cudaEventRecord(start, 0);
	
	//Add <<<nBloques,hilosB>>>(dev_vector1, dev_vector2, dev_resultado);
	invierte<<<1, N >>>(dev_vector1, dev_resultado);
	// cambiar (N,1) para n bloques de 1 hilo
	check_CUDA_Error("Error Kernel");
	
	// marca de final
	cudaEventRecord(stop, 0);
	// sincronizacion GPU-CPU
	cudaEventSynchronize(stop);
	// calculo del tiempo en milisegundos
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	//MEDICION DE TIEMPO EN GPU
	
	//recogida de los datos
	printf("\n");
	printf("vector de regreso:\n");
	cudaMemcpy(resultado, dev_resultado, N * sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error CudaMemcpy2");
	for (int i = 0; i < N; i++) {
		printf("%.2f, ", resultado[i]);

	}
	// impresion de resultados
	printf("\n");
	printf("> Tiempo de ejecucion: %f ms\n", elapsedTime);

	return 0;

	cudaFree(dev_vector1);
	cudaFree(dev_resultado);
	free(vector1);
	free(resultado);

}






/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/



#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 24 //tamano de los vectores
#define BLOCKS 6 // tamano del bloque(numero de hilos en cada bloque)


/*
suma de vectores de 3 dim, mide el tiempo de ejecucion
*/


// gridDim.x: La primera nos da el n�mero de bloques (M)
// blockDim.x: la segunda el n�mero de hilos que tiene cada bloque (N)



//Global: funcion llamada desde el host y ejecutada en el device(kernel)

__global__ void Add(float *a, float *b, float *c)
{
	int Id = threadIdx.x + blockDim.x * blockIdx.x;
	if (Id < N) {
		a[Id] = threadIdx.x;
		b[Id] = blockIdx.x;
		c[Id] = Id;
	}
}



// funcion para revision de errores en las funciones de CUDA

__host__ void check_CUDA_Error(const char *mensaje) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);	
	}
}



int main(int argc, char **argv)
{
	float *resultado1, *resultado2, *resultado3;
	float *resultado11, *resultado12, *resultado13;
	float *resultado21, *resultado22, *resultado23;

	float *dev_vector1, *dev_vector2, *dev_vector3;
	float *dev_vector11, *dev_vector12, *dev_vector13;
	float *dev_vector21, *dev_vector22, *dev_vector23;

	//reserva de memoria en el host
	resultado1 = (float*)malloc(N * sizeof(float));
	resultado2 = (float*)malloc(N * sizeof(float));
	resultado3 = (float*)malloc(N * sizeof(float));
	//reserva de memoria en el host
	resultado11 = (float*)malloc(N * sizeof(float));
	resultado12 = (float*)malloc(N * sizeof(float));
	resultado13 = (float*)malloc(N * sizeof(float));
	//reserva de memoria en el host
	resultado21 = (float*)malloc(N * sizeof(float));
	resultado22 = (float*)malloc(N * sizeof(float));
	resultado23 = (float*)malloc(N * sizeof(float));

	cudaError_t error;
	//reserva de memoria en el device
	error = cudaMalloc((void**)&dev_vector1, N * sizeof(float));
	error = cudaMalloc((void**)&dev_vector2, N * sizeof(float));
	error = cudaMalloc((void**)&dev_vector3, N * sizeof(float));

	error = cudaMalloc((void**)&dev_vector11, N * sizeof(float));
	error = cudaMalloc((void**)&dev_vector12, N * sizeof(float));
	error = cudaMalloc((void**)&dev_vector13, N * sizeof(float));

	error = cudaMalloc((void**)&dev_vector21, N * sizeof(float));
	error = cudaMalloc((void**)&dev_vector22, N * sizeof(float));
	error = cudaMalloc((void**)&dev_vector23, N * sizeof(float));

	if (error != cudaSuccess) {
		printf("\n ocurrio un error: %s", cudaGetErrorString(error));
	}

	//lanzamiento del kernel
	//calculamos el numero de bloques necesario para un tamano de bloque fijo 
	int nBloques = N / BLOCKS;
	if (N % BLOCKS != 0) {
		nBloques = nBloques + 1;
	}

	int nBloques2 = 1;
	int hilosB = BLOCKS;
	int hilosB2 = 1;


	// declaracion de eventos para medir el tiempo de ejecucion en la GPU
	cudaEvent_t start;
	cudaEvent_t stop;
	// creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// marca de inicio
	cudaEventRecord(start, 0);

	// codigo a temporizar en el device
	//ejecucion del kernel
	Add <<<nBloques, hilosB>>>(dev_vector1, dev_vector2, dev_vector3);
	check_CUDA_Error("Error Kernel 1");
	Add <<<nBloques2, N>>>(dev_vector11, dev_vector12, dev_vector13);
	check_CUDA_Error("Error Kernel 2");
	Add <<<N, 1>>>(dev_vector21, dev_vector22, dev_vector23);
	check_CUDA_Error("Error Kernel 3");
	// marca de final
	cudaEventRecord(stop, 0);
	// sincronizacion GPU-CPU
	cudaEventSynchronize(stop);
	// calculo del tiempo en milisegundos
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	

	//recogida de los datos
	cudaMemcpy(resultado1, dev_vector1, N * sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error dev_vector1");
	cudaMemcpy(resultado2, dev_vector2, N * sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error dev_vector2");
	cudaMemcpy(resultado3, dev_vector3, N * sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error dev_vector3");
	cudaMemcpy(resultado11, dev_vector11, N * sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error dev_vector11");
	cudaMemcpy(resultado12, dev_vector12, N * sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error dev_vector12");
	cudaMemcpy(resultado13, dev_vector13, N * sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error dev_vector13");
	cudaMemcpy(resultado21, dev_vector21, N * sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error dev_vector21");
	cudaMemcpy(resultado22, dev_vector22, N * sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error dev_vector22");
	cudaMemcpy(resultado23, dev_vector23, N * sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error dev_vector23");

	//impresion de los datos
	printf("\n");
	printf("vector de %d elementos\n", N);
	printf("Lanzamiento con %d bloques y %d hilos en cada bloque (%d hilos)\n", nBloques, BLOCKS,nBloques*hilosB);
	printf(">indice de hilo: \n");
	for (int i = 0; i < N; i++) {
		printf("%.0f, ", resultado1[i]);
	}
	printf("\n");
	printf(">indice de bloque: \n");
	for (int i = 0; i < N; i++) {
		printf("%.0f, ", resultado2[i]);
	}
	printf("\n");
	printf(">indice global: \n");
	for (int i = 0; i < N; i++) {
		printf("%.0f, ", resultado3[i]);
	}
	
	//
	printf("\n");
	printf("\n");
	printf("vector de %d elementos\n", N);
	printf("Lanzamiento con %d bloques (%d hilos)\n", nBloques2, nBloques*hilosB);

	//impresion de los datos
	printf(">indice de hilo: \n");
	for (int i = 0; i < N; i++) {
		printf("%.0f, ", resultado11[i]);
	}
	printf("\n");
	printf(">indice de bloque: \n");
	for (int i = 0; i < N; i++) {
		printf("%.0f, ", resultado12[i]);
	}
	printf("\n");
	printf(">indice global: \n");
	for (int i = 0; i < N; i++) {
		printf("%.0f, ", resultado13[i]);
	}

	//
	printf("\n");
	printf("\n");
	printf("vector de %d elementos\n", N);
	printf("Lanzamiento con %d bloques (%d hilos)\n", N, hilosB2);

	//impresion de los datos
	printf(">indice de hilo: \n");
	for (int i = 0; i < N; i++) {
		printf("%.0f, ", resultado21[i]);
	}
	printf("\n");
	printf(">indice de bloque: \n");
	for (int i = 0; i < N; i++) {
		printf("%.0f, ", resultado22[i]);
	}
	printf("\n");
	printf(">indice global: \n");
	for (int i = 0; i < N; i++) {
		printf("%.0f, ", resultado23[i]);
	}

	printf("\n");
	printf("\n");

	// impresion de resultados
	printf("> Tiempo de ejecucion: %f ms\n", elapsedTime);
	
	//liberacion de memoria del device
	cudaFree(dev_vector1);
	cudaFree(dev_vector2);
	cudaFree(dev_vector3);
	cudaFree(dev_vector11);
	cudaFree(dev_vector12);
	cudaFree(dev_vector13);
	cudaFree(dev_vector21);
	cudaFree(dev_vector22);
	cudaFree(dev_vector23);

	//liberacion de memoria del host
	free(resultado1);
	free(resultado2);
	free(resultado3);
	free(resultado11);
	free(resultado12);
	free(resultado13);
	free(resultado21);
	free(resultado22);
	free(resultado23);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("\n...");
	fflush(stdin);
	char tecla = getchar();
	
	return 0;

}







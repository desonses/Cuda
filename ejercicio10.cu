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

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define N 32

/*
calcula el valor aproximado de pi, realizando particiones (entre mas, hay mas aproximacion al valor)
*/

__host__ float func(float valor) 
{
	return  4 / (1 + powf(valor,2));
}

__global__ void calcula(float *particion, float *funcion, float *sum)
{
	//reserva dinamica de memoria compartida en tiempo de ejecucion
	extern __shared__ float temporal[];
	float add[N];
	//float h = (1 - 0) / N;
	int id = threadIdx.x;// +blockIdx.x * blockDim.x;
	
	float xi, xim; 
	float yi, yim;

	//printf("%.2f, \n", particion[id]);
	xi = particion[id];
	xim = particion[id - 1];
	yi = funcion[id];
	yim = funcion[id - 1];
	add[id] = .5f * ((xi - xim) * (yi + yim));
	temporal[id] = add[id];

	printf("(%.4f - %.4f) * (%.4f + %.4f): %.4f\n", xi, xim, yi, yim, temporal[id]);
	
	cuda_SYNCTHREADS();
	//reduccion paralela
	int salto = N / 2;
	
	//realizamos log2(N) iteraciones
	while (salto)
	{
		//solo trabajan la mitad de los hilos
		if (id < salto)
		{
			temporal[id] = temporal[id] + temporal[id + salto];
		}
		//cuda_SYNCTHREADS();
		cuda_SYNCTHREADS();
		salto = salto / 2;
	}
	//el hilo 0 escribe el resultado final en la  memoria global
	if (id == 0)
	{
		*sum = temporal[id];
		//printf("temporal: %.3f\n", *sum);
	}


}

int main(int argc, char** argv)
{
	float *vector1, *vector2, *resultado;
	float *dev_vector1, *dev_vector2, *dev_resultado;
	size_t size = N * sizeof(float);

	//reserva de memoria en el host
	vector1 = (float*)malloc(size);
	vector2 = (float*)malloc(size);
	resultado = (float*)malloc(size);

	//reserva de memoria en el device
	cudaMalloc((void**)&dev_vector1, size);
	cudaMalloc((void**)&dev_vector2, size);
	cudaMalloc((void**)&dev_resultado, size);

	// inicializacion de los vectores
	for (int i = 0; i < N; i++) {
		vector1[i] = (float)i / (N - 1);
		vector2[i] = func(vector1[i]);
		//printf("xi: %.2f, f(xi): %.2f \n", vector1[i], vector2[i]);
	}


	//enviar los datos hacia el Device
	cudaMemcpy(dev_vector1, vector1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vector2, vector2, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_resultado, resultado, size, cudaMemcpyHostToDevice);

	//lanzamiento del kernel con memoria dinamica compartida
	calcula <<<1, N, size>>>(dev_vector1, dev_vector2, dev_resultado);

	//recogida de los datos

	cudaMemcpy(resultado, dev_resultado, size, cudaMemcpyDeviceToHost);
	printf("pi = %.5f, \n", resultado[0]);

	return 0;
}

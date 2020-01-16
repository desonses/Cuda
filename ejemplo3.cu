
/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/



#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda_fp16.h>

/*
En este ejercicio se implemente un prog para resolver ecuaciones de segundo grado. 

*/

// Device: kernel que se ejecuta en la GPU
__global__ void suma_GPU(int a, int b, int *c)
{
	*c = a + b;
}

// solve equation second grade
__global__ void solve_GPU(int a, int b, int c ,int *x1, int *x2)
{
	int raiz = powf(b, 2) - (4 * a * c);
	int i = -b / 2 * a;
	int j = 2 * a;

	*x1 = i + sqrtf(raiz) / j;
	*x2 = i - sqrtf(raiz) / j;
}


// HOST: funcion llamada y ejecutada desde el host
__host__ int suma_CPU(int a, int b)
{
	return (a + b);
}


int main(int argc, char** argv)
{

	// declaraciones
	int n1 = 1, n2 = 2, c = 0;
	int *hst_c;

	int *hst_x1;
	int *hst_x2;

	int m1 = 10, m2 = 20;
	int *dev_c;

	// equacion
	int a = 1, b =8 , C = -6;
	int *dev_x1;
	int *dev_x2;

	// reserva de memoria en el host
	//hst_c = (int*)malloc( sizeof(int) );
	
	hst_x1 = (int*)malloc(sizeof(int));
	hst_x2 = (int*)malloc(sizeof(int));


	// reserva de memoria en el device
	//cudaMalloc((void**)&dev_c, sizeof(int) );

	cudaMalloc((void**)&dev_x1, sizeof(int));
	cudaMalloc((void**)&dev_x2, sizeof(int));



	// llamada a la funcion suma_CPU
	//c = suma_CPU(n1, n2);

	// resultados CPU
	//printf("CPU:\n");
	//printf("%2d + %2d = %2d \n",n1, n2, c);

	// llamada a la funcion suma_GPU
	//suma_GPU<<<1,1>>>(m1, m2, dev_c);

	solve_GPU<<<1,1>>>(a,b,C, dev_x1, dev_x2);

	// recogida de datos desde el device hacia el host
	//cudaMemcpy(hst_c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );

	cudaMemcpy(hst_x1, dev_x1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_x2, dev_x2, sizeof(int), cudaMemcpyDeviceToHost);

	printf("resultado: \n");
	printf("x1 = %2d ,x2 = %2d \n", *hst_x1, *hst_x2);

	// resultados GPU
	//printf("GPU:\n");
	//printf("%2d + %2d = %2d \n",m1, m2, *hst_c);

	// salida
	printf("\npulsa INTRO para finalizar...");
	fflush(stdin);
	char tecla = getchar();

	free(hst_c);//liberacion de memoria del host
	cudaFree(dev_c);//liberacion de memoria del device(kernel)
	return 0;

}



/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cpu_bitmap.h"

// Defines
#define DIM 1024 // Dimensiones del Bitmap

/*
generacion de una imagen en RGBa
*/

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)

__global__ void kernel(unsigned char *imagen)
{
	// coordenada horizontal
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	// coordenada vertical
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// coordenada global de cada pixel
	int pixel = x + y * blockDim.x * gridDim.x;
	
	// cada hilo pinta un pixel con un color arbitrario

	imagen[pixel * 4 + 0] = 255 * x / (blockDim.x * gridDim.x/8); // canal R
	imagen[pixel * 4 + 1] = 255 * y / (blockDim.y * gridDim.y/8); // canal G
	imagen[pixel * 4 + 2] = 2 * blockIdx.x + 2 * blockIdx.y/8; // canal B
	imagen[pixel * 4 + 3] = 255; // canal alfa

}



// MAIN: rutina principal ejecutada en el host

int main(int argc, char** argv)
{
	// declaracion del bitmap
	CPUBitmap bitmap(DIM, DIM);

	// tama�o en bytes
	size_t size = bitmap.image_size();
	
	// reserva en el host
	unsigned char *host_bitmap = bitmap.get_ptr();
	
	// reserva en el device
	unsigned char *dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, size);
	
	// generamos el bitmap
	dim3 Nbloques(DIM / 16, DIM / 16);
	dim3 hilosB(16, 16);
	kernel <<<Nbloques, hilosB >>> (dev_bitmap);
	
	// recogemos el bitmap desde la GPU para visualizarlo
	cudaMemcpy(host_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost);
	
	// liberacion de recursos
	cudaFree(dev_bitmap);
	
	// visualizacion y salida
	printf("\n...pulsa ESC para finalizar...");
	bitmap.display_and_exit();
	return 0;
}



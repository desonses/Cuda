/*
autor fredy m
uaem
desonses@gmail.com para mas comentarios
*/

#include "cpu_bitmap.h" 
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define Nx 1920/2
#define Ny 1080/2

/*
si muestra el error LNK1104 glut64.lib, se debe colocar el archivo en la
carpeta de cuda: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64
revisar, la direccion es similar, algunas que pueden generar problema son
cpu_bitmap.h
*/


struct cuComplex {
	float   r;
	float   i;
	// cuComplex( float a, float b ) : r(a), i(b)  {}
	__device__ cuComplex(float a, float b) : r(a), i(b) {} // Fix error for calling host function from device
	__device__ float magnitude2(void) 
	{
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a) 
	{
		return cuComplex(r*a.r - i * a.i, i*a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) 
	{
		return cuComplex(r + a.r, i + a.i);
	}
	
};


__device__ int julia(int x, int y) {
	const float scale = 1.5;
	float jx = scale * (float)(Nx / 2 - x) / (Nx / 2);
	float jy = scale * (float)(Ny / 2 - y) / (Ny / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}



__global__ void change_bitmap(unsigned char *ptr) {
	// map from blockIdx to pixel position
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	// now calculate the value at that position
	int juliaValue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main(int argc, char const *argv[]) {

	CPUBitmap host_bitmap(Nx, Ny);
	unsigned char *device_bitmap;

	cudaMalloc((void **)&device_bitmap, host_bitmap.image_size());

	dim3 grid(Nx, Ny);
	change_bitmap <<<grid, 1 >>>(device_bitmap);
	cudaMemcpy(host_bitmap.get_ptr(), device_bitmap, host_bitmap.image_size(), cudaMemcpyDeviceToHost);
	cudaFree(device_bitmap);

	host_bitmap.display_and_exit();

	return 0;
}

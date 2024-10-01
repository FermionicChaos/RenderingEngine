#include "cVecR4.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cVecR4::cVecR4() {
	cudaMalloc(&d_ptr, 4*sizeof(double));
	h_ptr = new double[4];
}

cVecR4::~cVecR4() {
	cudaFree(d_ptr);
	delete[] h_ptr;
}

cVecR4::cVecR4(cVecR4 & inp) {
	cudaMalloc(&d_ptr, 4*sizeof(double));
	h_ptr = new double[4];
	cudaMemcpy(d_ptr, inp.d_ptr, 4*sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(h_ptr, d_ptr, 4 * sizeof(double), cudaMemcpyDeviceToHost);
	metric = inp.metric;
	vec = inp.vec;
}

cVecR4 & cVecR4::operator=(cVecR4 & rhs) {
	cudaMalloc(&(this->d_ptr), 4 * sizeof(double));
	this->h_ptr = new double[4];
	cudaMemcpy(this->d_ptr, rhs.d_ptr, 4 * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(this->h_ptr, this->d_ptr, 4 * sizeof(double), cudaMemcpyDeviceToHost);
	this->metric = rhs.metric;
	this->vec = rhs.vec;
	return *this;
}

cVecR4::cVecR4(cVecR4 && inp) {
	cudaMalloc(&d_ptr, 4 * sizeof(double));
	h_ptr = new double[4];
	cudaMemcpy(d_ptr, inp.d_ptr, 4 * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(h_ptr, d_ptr, 4 * sizeof(double), cudaMemcpyDeviceToHost);
	metric = inp.metric;
	vec = inp.vec;

	cudaFree(inp.d_ptr);
	delete[] h_ptr;
}

cVecR4 & cVecR4::operator=(cVecR4 && rhs) {
	cudaMalloc(&(this->d_ptr), 4 * sizeof(double));
	this->h_ptr = new double[4];
	cudaMemcpy(this->d_ptr, rhs.d_ptr, 4 * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(this->h_ptr, rhs.d_ptr, 4 * sizeof(double), cudaMemcpyDeviceToHost);
	this->metric = rhs.metric;
	this->vec = rhs.vec;

	cudaFree(rhs.d_ptr);
	delete[] h_ptr;
	return *this;
}

cVecR4::cVecR4(vec_t v_type, metric_t m_type, double z0, double z1, double z2, double z3) {
	vec = v_type;
	metric = m_type;

	cudaMalloc(&d_ptr, 4*sizeof(double));
	h_ptr = new double[4];
	h_ptr[0] = z0;
	h_ptr[1] = z1;
	h_ptr[2] = z2;
	h_ptr[3] = z3;

	cudaMemcpy(d_ptr, h_ptr, 4*sizeof(double), cudaMemcpyHostToDevice);
}

double cVecR4::operator()(int i) {
	cudaMemcpy(h_ptr, d_ptr, 4 * sizeof(double), cudaMemcpyDeviceToHost);
	return h_ptr[i];
}

void cVecR4::send2Device() {
	cudaMemcpy(d_ptr, h_ptr, 4 * sizeof(double), cudaMemcpyHostToDevice);
}

void cVecR4::call2Host() {
	cudaMemcpy(h_ptr, d_ptr, 4 * sizeof(double), cudaMemcpyDeviceToHost);
}

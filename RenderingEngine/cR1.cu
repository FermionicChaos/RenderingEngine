#include "cR1.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gsl\gsl_poly.h"

#include "stdud.h"
using namespace std;
/*

*/
__global__ void cR1_Copy(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = a[i];
	}

}

__global__ void cuda_Add(double* c, double* a, double* b, int N) {
	//int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] + b[i];
	}
}

__global__ void cuda_Sub(double* c, double* a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] - b[i];
	}
}

__global__ void cuda_Mult(double* c, double* a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] * b[i];
	}
}

__global__ void cuda_Divide(double* c, double* a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		if (b[i] != 0.0) {
			c[i] = a[i] / b[i];
		}
		else {
			c[i] = 0.0;
		}
	}
}

__global__ void cuda_Add_RHS(double* c, double* a, double b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] + b;
	}
}

__global__ void cuda_Sub_RHS(double* c, double* a, double b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] - b;
	}
}

__global__ void cuda_Mult_RHS(double* c, double* a, double b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] * b;
	}
}

__global__ void cuda_Divide_RHS(double* c, double* a, double b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] / b;
	}
}

__global__ void cuda_Add_LHS(double* c, double a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a + b[i];
	}
}

__global__ void cuda_Sub_LHS(double* c, double a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a - b[i];
	}
}

__global__ void cuda_Mult_LHS(double* c, double a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a * b[i];
	}
}

__global__ void cuda_Divide_LHS(double* c, double a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		if (b[i] != 0.0) {
			c[i] = a / b[i];
		}
		else {
			c[i] = 0.0;
		}
	}
}
/*

*/
cR1::cR1() {
	//Default Constructor
	h_ptr = nullptr;
	m = 0; dx = 0; x1 = 0; x2 = 0;
	grid = 0; block = 0;
	//cout << "Null Constructor: ADDRESS -> " << base << endl;
}

cR1::~cR1() {
	delete[] h_ptr;
	cudaFree(&d_ptr);
}

cR1::cR1(const cR1 & inp) {
	//Copy Construct
	dx = inp.getDX(); m = inp.getSize();
	x1 = inp.getX1(); x2 = inp.getX2();
	grid = inp.getG(); block = inp.getB();
	h_ptr = new double[m];
	cudaMalloc(&d_ptr, inp.getMem());
	cR1_Copy << <grid, block >> > (d_ptr, inp.getD_ptr(), m);
}

cR1 & cR1::operator=(const cR1 & rhs) {
	//Copy Assign
	if ((h_ptr != rhs.getH_ptr())&&(d_ptr != rhs.getD_ptr())&&(m == rhs.getSize())) {
		x1 = rhs.getX1(); x2 = rhs.getX2();
		grid = rhs.getG(); block = rhs.getB();
		cR1_Copy << <grid, block >> > (d_ptr, rhs.getD_ptr(), rhs.getSize());
		return *this;
	}
	else if ((h_ptr != rhs.getH_ptr()) && (d_ptr != rhs.getD_ptr())) {
		x1 = rhs.getX1(); x2 = rhs.getX2();
		grid = rhs.getG(); block = rhs.getB();
		m = rhs.getSize();
		delete[] this->h_ptr;
		cudaFree(d_ptr);
		this->h_ptr = new double[rhs.getSize()];
		cudaMalloc(&d_ptr, rhs.getMem());

		cR1_Copy << <grid, block >> > (d_ptr, rhs.getD_ptr(), rhs.getSize());

		return *this;
	}
	else {
		return *this;
	}
}

cR1::cR1(cR1 && inp) {
	//Move Construct
	x1 = inp.getX1(); x2 = inp.getX2();
	grid = inp.getG(); block = inp.getB();
	m = inp.getSize();

	h_ptr = inp.h_ptr;
	inp.h_ptr = nullptr;
	d_ptr = inp.d_ptr;
	inp.d_ptr = nullptr;
}

cR1 & cR1::operator=(cR1 && rhs) {
	//Move Assign
	x1 = rhs.getX1(); x2 = rhs.getX2();
	grid = rhs.getG(); block = rhs.getB();
	m = rhs.getSize();

	delete[] h_ptr;
	h_ptr = rhs.h_ptr;
	cudaFree(d_ptr);
	d_ptr = rhs.d_ptr;
	rhs.h_ptr = nullptr;
	rhs.d_ptr = nullptr;
	return *this;
}

cR1::cR1(int size) {
	h_ptr = new double[size];
	cudaMalloc(&d_ptr, size * sizeof(double));
	m = size; dx = 0; x1 = 0; x2 = 0;
}

cR1::cR1(double a, double b, int size) {
	//Generate Number line
	m = size; x1 = a; x2 = b;
	double N = static_cast<double>(size);
	dx = (b - a) / N;
	double j = 0;
	double* temp = new double[m];
	for (int i = 0; i < m; ++i) {
		temp[i] = a + j*dx;
		j = j + 1;
	}
	h_ptr = temp;
	cudaMalloc(&d_ptr, m*sizeof(double));
	cudaMemcpy(d_ptr, h_ptr, m * sizeof(double), cudaMemcpyHostToDevice);
}

double cR1::operator()(int I) {
	if ((I >= 0) && (I <= m)) {
		return h_ptr[I];
	}
	else {
		cout << "Not an element" << endl;
		return 0.0;
	}
}

double cR1::operator()(double p) {
	double I0;
	int I;
	I0 = (p - x1) / dx;
	I = static_cast<int>(I0);
	if ((I >= 0) && (I <= m)) {
		return h_ptr[I];
	}
	else {
		cout << "Not an element" << endl;
		return 0.0;
	}
}

cR1 cR1::operator+(const cR1 & rhs) {
	cR1 temp(rhs.getSize());
	if (this->sameDim(rhs)) {
		temp.setDX(dx); temp.setSize(m);
		temp.setX1(x1); temp.setX2(x2);
		temp.setG(grid); temp.setB(block);
		cuda_Add << <grid, block >> > (temp.getD_ptr(), d_ptr, rhs.getD_ptr(), m);
		return temp;
	}
	else {
		cout << "Error: Dimensions must agree!" << endl;
		return temp;
	}
}

cR1 cR1::operator-(const cR1 & rhs) {
	cR1 temp(rhs.getSize());
	if (this->sameDim(rhs)) {
		temp.setDX(dx); temp.setSize(m);
		temp.setX1(x1); temp.setX2(x2);
		temp.setG(grid); temp.setB(block);
		cuda_Sub << <grid, block >> > (temp.getD_ptr(), d_ptr, rhs.getD_ptr(), m);
		return temp;
	}
	else {
		cout << "Error: Dimensions must agree!" << endl;
		return temp;
	}
}

cR1 cR1::operator*(const cR1 & rhs) {
	cR1 temp(rhs.getSize());
	if (this->sameDim(rhs)) {
		temp.setDX(dx); temp.setSize(m);
		temp.setX1(x1); temp.setX2(x2);
		temp.setG(grid); temp.setB(block);
		cuda_Mult << <grid, block >> > (temp.getD_ptr(), d_ptr, rhs.getD_ptr(), m);
		return temp;
	}
	else {
		cout << "Error: Dimensions must agree!" << endl;
		return temp;
	}
}

cR1 cR1::operator/(const cR1 & rhs) {
	cR1 temp(rhs.getSize());
	if (this->sameDim(rhs)) {
		temp.setDX(dx); temp.setSize(m);
		temp.setX1(x1); temp.setX2(x2);
		temp.setG(grid); temp.setB(block);
		cuda_Divide << <grid, block >> > (temp.getD_ptr(), d_ptr, rhs.getD_ptr(), m);
		return temp;
	}
	else {
		cout << "Error: Dimensions must agree!" << endl;
		return temp;
	}
}

cR1 operator+(const cR1 & lhs, double rhs) {
	cR1 temp(lhs.getSize());

	temp.setX1(lhs.getX1()); temp.setX2(lhs.getX2());
	temp.setDX(lhs.getDX()); temp.setSize(lhs.getSize());
	temp.setG(lhs.getG()); temp.setB(lhs.getB());

	cuda_Add_RHS << <lhs.getG(),lhs.getB() >> > (temp.getD_ptr(), lhs.getD_ptr(), rhs, lhs.getSize());
	return temp;
}

cR1 operator-(const cR1 & lhs, double rhs) {
	cR1 temp(lhs.getSize());

	temp.setX1(lhs.getX1()); temp.setX2(lhs.getX2());
	temp.setDX(lhs.getDX()); temp.setSize(lhs.getSize());
	temp.setG(lhs.getG()); temp.setB(lhs.getB());

	cuda_Sub_RHS << <lhs.getG(), lhs.getB() >> > (temp.getD_ptr(), lhs.getD_ptr(), rhs, lhs.getSize());
	return temp;
}

cR1 operator*(const cR1 & lhs, double rhs) {
	cR1 temp(lhs.getSize());

	temp.setX1(lhs.getX1()); temp.setX2(lhs.getX2());
	temp.setDX(lhs.getDX()); temp.setSize(lhs.getSize());
	temp.setG(lhs.getG()); temp.setB(lhs.getB());

	cuda_Mult_RHS << <lhs.getG(), lhs.getB() >> > (temp.getD_ptr(), lhs.getD_ptr(), rhs, lhs.getSize());
	return temp;
}

cR1 operator/(const cR1 & lhs, double rhs) {
	if (rhs != 0.0) {
		cR1 temp(lhs.getSize());

		temp.setX1(lhs.getX1()); temp.setX2(lhs.getX2());
		temp.setDX(lhs.getDX()); temp.setSize(lhs.getSize());
		temp.setG(lhs.getG()); temp.setB(lhs.getB());

		cuda_Divide_RHS << <lhs.getG(), lhs.getB() >> > (temp.getD_ptr(), lhs.getD_ptr(), rhs, lhs.getSize());
		return temp;
	}
	else {
		cout << "Division by zero is not Allowed!" << endl;
		return lhs;
	}
}
//Son of a fuckiung bitch! I forgot friend functions have access! hfiohdhg!@$R%@TET
cR1 operator+(double lhs, const cR1 & rhs) {
	cR1 temp(rhs.getSize());

	temp.setX1(rhs.getX1()); temp.setX2(rhs.getX2());
	temp.setDX(rhs.getDX()); temp.setSize(rhs.getSize());
	temp.setG(rhs.getG()); temp.setB(rhs.getB());

	cuda_Add_LHS << <rhs.getG(), rhs.getB() >> > (temp.getD_ptr(), lhs, rhs.getD_ptr(), rhs.getSize());
	return temp;
}

cR1 operator-(double lhs, const cR1 & rhs) {
	cR1 temp(rhs.getSize());

	temp.setX1(rhs.getX1()); temp.setX2(rhs.getX2());
	temp.setDX(rhs.getDX()); temp.setSize(rhs.getSize());
	temp.setG(rhs.getG()); temp.setB(rhs.getB());

	cuda_Sub_LHS << <rhs.getG(), rhs.getB() >> > (temp.getD_ptr(), lhs, rhs.getD_ptr(), rhs.getSize());
	return temp;
}

cR1 operator*(double lhs, const cR1 & rhs) {
	cR1 temp(rhs.getSize());

	temp.setX1(rhs.getX1()); temp.setX2(rhs.getX2());
	temp.setDX(rhs.getDX()); temp.setSize(rhs.getSize());
	temp.setG(rhs.getG()); temp.setB(rhs.getB());

	cuda_Mult_LHS << <rhs.getG(), rhs.getB() >> > (temp.getD_ptr(), lhs, rhs.getD_ptr(), rhs.getSize());
	return temp;
}

cR1 operator/(double lhs, const cR1 & rhs) {
	cR1 temp(rhs.getSize());

	temp.setX1(rhs.getX1()); temp.setX2(rhs.getX2());
	temp.setDX(rhs.getDX()); temp.setSize(rhs.getSize());
	temp.setG(rhs.getG()); temp.setB(rhs.getB());

	cuda_Divide_LHS << <rhs.getG(), rhs.getB() >> > (temp.getD_ptr(), lhs, rhs.getD_ptr(), rhs.getSize());
	return temp;
}

cR1 d_dx(const cR1 & f, int Ver) {
	//Numerical derivative of variance {1 2 3}.
	cR1 temp(f);
	temp.call2Host();
	int N = f.getSize();

	if (Ver == 1) { //Alg #1
		for (int i = 0; i < (N - 1); ++i) {
			temp.h_ptr[i] = (f.h_ptr[i + 1] - f.h_ptr[i]) / (f.dx);
		}
		temp.h_ptr[N - 1] = 0.75*temp.h_ptr[N - 2];
		temp.send2Device();
		return temp;
	}
	else if (Ver == 2) { //Alg #2
		for (int i = 1; i < (N - 1); ++i) {
			temp.h_ptr[i] = (f.h_ptr[i + 1] - f.h_ptr[i - 1]) / (2 * f.dx);
		}
		temp.h_ptr[0] = 0.25*temp.h_ptr[1];
		temp.h_ptr[N - 1] = 0.5*temp.h_ptr[N - 2];
		temp.send2Device();
		return temp;
	}
	else if (Ver == 3) { // Alg# 3
		for (int i = 2; i < (N - 2); ++i) {
			temp.h_ptr[i] = 2 * (f.h_ptr[i + 1] - f.h_ptr[i - 1]) / (3 * f.dx) -
				(f.h_ptr[i + 2] - f.h_ptr[i - 2]) / (12 * f.dx);
		}
		temp.h_ptr[1] = 0.5*temp.h_ptr[2];
		temp.h_ptr[0] = 0.25*temp.h_ptr[1];
		temp.h_ptr[N - 2] = 0.5*temp.h_ptr[N - 3];
		temp.h_ptr[N - 1] = 0.5*temp.h_ptr[N - 2];
		temp.send2Device();
		return temp;
	}
	else {
		cout << "INVALID ORDER!" << endl;
		return temp;
	}
}

double integralD(double a, double b, const cR1 & f) {
	cR1* temp = new cR1(f);
	temp->call2Host();
	int N = f.getSize();
	int N1 = temp->getIndex(a);
	int N2 = temp->getIndex(b);
	double Sum = 0.5*(f.h_ptr[N1] + f.h_ptr[N2])*f.dx;
	for (int i = N1 + 1; i < (N2 - 1); ++i) {
		Sum = Sum + f.h_ptr[i] * f.dx;
	}
	delete temp;
	return Sum;
}

int cR1::getIndex(double p) {
	//Find index of value in set based on domain.
	if ((p >= x1) && (p <= x2)) {
		double N = (p - x1) / dx;
		int N2 = static_cast<int>(N);
		return h_ptr[N2];
	}
	else {
		cout << "NOT AN ELEMENT!" << endl;
		return 0.0;
	}
}

void cR1::send2Device() {
	cudaMemcpy(d_ptr, h_ptr, m * sizeof(double), cudaMemcpyHostToDevice);
}

void cR1::call2Host() {
	cudaMemcpy(h_ptr, d_ptr, m * sizeof(double), cudaMemcpyDeviceToHost);
}

bool cR1::sameDim(const cR1 & rhs) {
	if ((m == rhs.getSize()) && (x1 == rhs.getX1()) && (x2 == rhs.getX2()) && (dx = rhs.getDX())) {
		return true;
	}
	else {
		return false;
	}
}

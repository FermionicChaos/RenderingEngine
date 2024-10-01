#include "cR2.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdud.h"
#include <algorithm>
#include "VecR2.h"
using namespace std;

/*
Allocation and transfer take the most amount of time! v2 is designed to keep object gpu dual
until the end of it's life. All computation will be done on the GPU for fast computation of the
Hydrogen atom and other quantum systems. It is still unknown whether or not this will work.
A realtime Simulator of the hydrogen atom is the goal with arbitrary E states and time evolution.
What will be needed is a state vector H |psi> = Ci*|n,l,m> transfer on GPU with, phase exp(-i*tt/n*n).
With the final computation on the gpu being Ci * exp(-i*tt/n*n) * Psi_nlm();.

Modify if you wish Memcpy to host, to increase runtime.
Should I only allocate space on host when I deploy call2Host()?
*/

__global__ void CUDA_Copy(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = a[i];
	}

}

__global__ void cR2_Add(double* c, double* a, double* b, int N) {
	//int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] + b[i];
	}
}

__global__ void cR2_Sub(double* c, double* a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] - b[i];
	}
}

__global__ void cR2_Mult(double* c, double* a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] * b[i];
	}
}

__global__ void cR2_Divide(double* c, double* a, double* b, int N) {
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

__global__ void cR2_Add_RHS(double* c, double* a, double b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] + b;
	}
}

__global__ void cR2_Sub_RHS(double* c, double* a, double b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] - b;
	}
}

__global__ void cR2_Mult_RHS(double* c, double* a, double b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] * b;
	}
}

__global__ void cR2_Divide_RHS(double* c, double* a, double b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] / b;
	}
}

__global__ void cR2_Add_LHS(double* c, double a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a + b[i];
	}
}

__global__ void cR2_Sub_LHS(double* c, double a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a - b[i];
	}
}

__global__ void cR2_Mult_LHS(double* c, double a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = a * b[i];
	}
}

__global__ void cR2_Divide_LHS(double* c, double a, double* b, int N) {
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

cR2::cR2() {
	h_ptr = nullptr;
	cudaMalloc(&d_ptr, sizeof(double));
	x1 = 0, x2 = 0, y1 = 0, y2 = 0,
		dx = 0, dy = 0;
	m = 0, n = 0;
	dim3 grid = 0, block = 0;
}
//Wurks
cR2::~cR2() {
	//Unknown if one should delete the pointer d_ptr. TEST ME FOR LEAKS
	delete[] h_ptr;
	cudaFree(d_ptr);
	//delete[] d_ptr;
}
//Wurks
cR2::cR2(const cR2 & rhs) {
	//Non r-values. Copy constructor
	dx = rhs.getDX(); dy = rhs.getDY();
	x1 = rhs.getX1(); x2 = rhs.getX2();
	y1 = rhs.getY1(); y2 = rhs.getY2();
	m = rhs.getIndex1(); n = rhs.getIndex2();
	grid = rhs.getG(); block = rhs.getB();
	h_ptr = new double[m*n]; //Allocate Host MEM
	cudaMalloc(&d_ptr, rhs.getMEM()); //Allocate Device MEM
	CUDA_Copy << <grid, block >> > (d_ptr, rhs.getD_ptr(), rhs.getSize());
	//cudaMemcpy(h_ptr, d_ptr, rhs.getMEM(), cudaMemcpyDeviceToHost);
}
//Wurks
cR2 & cR2::operator=(const cR2 & rhs) {
	// You MUST initialize and reserve on GPU to cut runtime! O.W. use Malloc.
	// i.e. (cR2 Func(M, N);)
	//cout << "Copy assign" << endl;
	if ((h_ptr != rhs.getH_ptr()) && (m == rhs.getIndex1()) && (n == rhs.getIndex2())) {
		//If dimensions match!
		//cout << "Dimensions agree" << endl;
		this->dx = rhs.getDX(); this->dy = rhs.getDY();
		this->x1 = rhs.getX1(); this->x2 = rhs.getX2();
		this->y1 = rhs.getY1(); this->y2 = rhs.getY2();
		//this->m = rhs.getIndex1(); this->n = rhs.getIndex2();
		this->grid = rhs.getG(); this->block = rhs.getB();
		delete[] this->h_ptr;
		//cudaFree(this->d_ptr);
		//cudaMalloc(&d_ptr, rhs.getMEM());
		this->h_ptr = new double[m*n];
		CUDA_Copy << <grid, block >> > (d_ptr, rhs.getD_ptr(), rhs.getSize());
		//cudaMemcpy(h_ptr, d_ptr, rhs.getMEM(), cudaMemcpyDeviceToHost);
		//this->h_ptr = h_ptr;
		return *this;
	}
	else if (h_ptr != rhs.getH_ptr()) {
		//cout << "Dimensions do not agree" << endl;
		//If dimensions do not match, allocate new memory on GPU
		this->dx = rhs.getDX(); this->dy = rhs.getDY();
		this->x1 = rhs.getX1(); this->x2 = rhs.getX2();
		this->y1 = rhs.getY1(); this->y2 = rhs.getY2();
		this->m = rhs.getIndex1(); this->n = rhs.getIndex2();
		this->grid = rhs.getG(); this->block = rhs.getB();
		delete[] this->h_ptr;
		cudaFree(this->d_ptr);
		cudaMalloc(&d_ptr, rhs.getMEM());
		this->h_ptr = new double[m*n];
		CUDA_Copy << <grid, block >> > (d_ptr, rhs.getD_ptr(), rhs.getSize());
		//cudaMemcpy(h_ptr, d_ptr, rhs.getMEM(), cudaMemcpyDeviceToHost);
		//this->h_ptr = h_ptr;
		return *this;
	}
	else {
		//cout << "No copy assign" << endl;
		return *this;
	}
}
//WARNING CHECK R-VALUES
cR2::cR2(cR2 && inp) {
	/* r-value constructor move pointers
	DO NOT CONSTRUCT FROM AN ALREADY CONSTRUCTED
	FUNCTION! i.e. function already created, cR2 F;
	F(x + y + ... 8==C>^2) If you do, may memory leaks
	haunt you and your children forever... */
	h_ptr = inp.h_ptr;
	inp.h_ptr = nullptr;
	d_ptr = inp.d_ptr;
	inp.d_ptr = nullptr;
	dx = inp.getDX(); dy = inp.getDY();
	x1 = inp.getX1(); x2 = inp.getX2();
	y1 = inp.getY1(); y2 = inp.getY2();
	m = inp.getIndex1(); n = inp.getIndex2();
	grid = inp.getG(); block = inp.getB();
}
//WARNING CHECK R-VALUES
cR2 & cR2::operator=(cR2 && rhs) {
	delete[] h_ptr;
	h_ptr = rhs.h_ptr;
	cudaFree(d_ptr);
	d_ptr = rhs.d_ptr;
	dx = rhs.dx; dy = rhs.dy;
	x1 = rhs.x1; x2 = rhs.x2;
	y1 = rhs.y1; y2 = rhs.y2;
	m = rhs.m; n = rhs.n;
	grid = rhs.grid; block = rhs.block;
	rhs.h_ptr = nullptr;
	rhs.d_ptr = nullptr;
	return *this;
}
//Wurks
cR2::cR2(int M, int N) {
	//It would be wise to allocate in advance for realtime functions,
	//but that is none of my business.
	m = M; n = N;
	dx = 0.0; dy = 0.0;
	x1 = 0.0; x2 = 0.0; y1 = 0.0; y2 = 0.0;
	grid = M*N / 1024; block = 1024;
	h_ptr = new double[M*N];
	cudaMalloc(&d_ptr, M*N * sizeof(double));
}
//Work on, still need to generate on GPU.
cR2::cR2(int k, double X1, double X2, int M, double Y1, double Y2, int N) {
	//Note2self: Test CUDA for R2 arrays and multi indexing.
	if ((X2>X1) && (Y2>Y1)) {
		m = M; n = N; x1 = X1; x2 = X2; y1 = Y1; y2 = Y2;
		h_ptr = new double[M*N];
		cudaMalloc(&d_ptr, M*N * sizeof(double));
		// FIX ROUND UP/DOWN CASTING
		// Set up for generating on GPU.
		grid = M*N / 1024; block = 1024;
		double M2 = double(M), N2 = double(N);
		dx = (X2 - X1) / M2; dy = (Y2 - Y1) / N2;
		if (k == 1) {
			//generate x-domain
			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < n; ++j) {
					h_ptr[i + j*m] = X1 + double(i)*dx;
				}
			}
			y2 = y1 + (n - 1)*dy; x2 = x1 + (m - 1)*dx;
			cudaMemcpy(d_ptr, h_ptr, M*N * sizeof(double), cudaMemcpyHostToDevice);
		}
		else if (k == 2) {
			//generate y-domain
			for (int i = 0; i < m; ++i) {
				double J = 0;
				for (int j = 0; j < n; ++j) {
					h_ptr[i + j*m] = y1 + J*dy;
					J += 1.0;
				}
			}
			y2 = y1 + (n - 1)*dy; x2 = x1 + (m - 1)*dx;
			cudaMemcpy(d_ptr, h_ptr, M*N * sizeof(double), cudaMemcpyHostToDevice);
		}
		else {
			cout << "YOU GET GARBAGE, FUCK YOU!" << endl;
		}
	}
	else {
		m = M; n = N;
		dx = 0.0; dy = 0.0;
		x1 = 0.0; x2 = 0.0; y1 = 0.0; y2 = 0.0;
		grid = 0; block = 0;
		h_ptr = new double[M*N];
	}
}

double cR2::operator()(int I, int J) {
	//DO NOT FORGET TO COPY TO HOST BEFORE THIS CALL!
	if ((I >= 0) && (J >= 0) && (I<m) && (J<n)) {
		return h_ptr[I + J*m];
	}
	else {
		//cout << "Not an element" << endl;
		return 0.0;
	}
}

double cR2::operator()(double X, double Y) {
	//DO NOT FORGET TO COPY TO HOST BEFORE THIS CALL!
	//int input;
	double I0, J0;
	int I, J;
	I0 = (X - x1) / dx;
	J0 = (Y - y1) / dy;
	//cout << "Cock elements" << I0 << " cocks " << J0 << endl;
	//cin >> input;
	I = static_cast<int>(I0);
	J = static_cast<int>(J0);
	//cout << "Cock elements" << I << " cocks " << J << "bock bock " << endl;
	//cin >> input;
	if ((I >= 0) && (J >= 0) && (I<m) && (J<n)) {
		return h_ptr[I + J*m];
	}
	else {
		cout << "Not an element!" << endl;
		return 0.0;
	}
}

double cR2::operator()(VecR2 & pos) {
	//DO NOT FORGET TO COPY TO HOST BEFORE THIS CALL!
	double I0 = (pos.getX() - x1) / dx, J0 = (pos.getY() - y1) / dy;
	int I = int(I0), J = int(J0);
	if ((I >= 0) && (J >= 0) && (I<m) && (J<n)) {
		return h_ptr[I + J*m];
	}
	else {
		cout << "Not an element" << endl;
		return 0.0;
	}
}

cR2 cR2::operator+(const cR2 & rhs) {
	cR2 temp(m, n);
	if (sameDim(*this, rhs)) {
		temp.setDX(dx); temp.setDY(dy);

		temp.setX1(x1); temp.setX2(x2);
		temp.setY1(y1); temp.setY2(y2);
		temp.setG(grid); temp.setB(block);
		//Execute GPU duals:
		cR2_Add << <grid, block >> >(temp.getD_ptr(), d_ptr, rhs.d_ptr, rhs.getSize());

		//Optional: COPY to host, find better place!
		//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
		return temp;
	}
	else {
		cout << "Error: Dimensions must agree!" << endl;
		return temp;
	}
}

cR2 cR2::operator-(const cR2 & rhs) {
	cR2 temp(m, n);
	if (sameDim(*this, rhs)) {
		temp.setDX(dx); temp.setDY(dy);

		temp.setX1(x1); temp.setX2(x2);
		temp.setY1(y1); temp.setY2(y2);
		temp.setG(grid); temp.setB(block);
		//Execute GPU duals:
		cR2_Sub << <grid, block >> >(temp.getD_ptr(), d_ptr, rhs.d_ptr, rhs.getSize());

		//Optional: COPY to host, find better place!
		//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
		return temp;
	}
	else {
		cout << "Error: Dimensions must agree!" << endl;
		return temp;
	}
}

cR2 cR2::operator*(const cR2 & rhs) {
	cR2 temp(m, n);
	if (sameDim(*this, rhs)) {
		temp.setDX(dx); temp.setDY(dy);

		temp.setX1(x1); temp.setX2(x2);
		temp.setY1(y1); temp.setY2(y2);
		temp.setG(grid); temp.setB(block);
		//Execute GPU duals:
		cR2_Mult << <grid, block >> >(temp.getD_ptr(), d_ptr, rhs.d_ptr, rhs.getSize());

		//Optional: COPY to host, find better place!
		//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
		return temp;
	}
	else {
		cout << "Error: Dimensions must agree!" << endl;
		return temp;
	}
}

cR2 cR2::operator/(const cR2 & rhs) {
	cR2 temp(m, n);
	if (sameDim(*this, rhs)) {
		temp.setDX(dx); temp.setDY(dy);

		temp.setX1(x1); temp.setX2(x2);
		temp.setY1(y1); temp.setY2(y2);
		temp.setG(grid); temp.setB(block);
		//Execute GPU duals:
		cR2_Divide << <grid, block >> >(temp.getD_ptr(), d_ptr, rhs.d_ptr, rhs.getSize());

		//Optional: COPY to host, find better place!
		//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
		return temp;
	}
	else {
		cout << "Error: Dimensions must agree!" << endl;
		return temp;
	}
}

cR2 operator+(const cR2 & lhs, double rhs) {
	cR2 temp(lhs.m, lhs.n);
	temp.setDX(lhs.dx); temp.setDY(lhs.dy);

	temp.setX1(lhs.x1); temp.setX2(lhs.x2);
	temp.setY1(lhs.y1); temp.setY2(lhs.y2);
	temp.setG(lhs.grid); temp.setB(lhs.block);
	//Execute GPU duals:
	cR2_Add_RHS << <lhs.grid, lhs.block >> >(temp.getD_ptr(), lhs.d_ptr, rhs, lhs.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 operator-(const cR2 & lhs, double rhs) {
	cR2 temp(lhs.m, lhs.n);
	temp.setDX(lhs.dx); temp.setDY(lhs.dy);

	temp.setX1(lhs.x1); temp.setX2(lhs.x2);
	temp.setY1(lhs.y1); temp.setY2(lhs.y2);
	temp.setG(lhs.grid); temp.setB(lhs.block);
	//Execute GPU duals:
	cR2_Sub_RHS << <lhs.grid, lhs.block >> >(temp.getD_ptr(), lhs.d_ptr, rhs, lhs.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 operator*(const cR2 & lhs, double rhs) {
	cR2 temp(lhs.m, lhs.n);
	temp.setDX(lhs.dx); temp.setDY(lhs.dy);

	temp.setX1(lhs.x1); temp.setX2(lhs.x2);
	temp.setY1(lhs.y1); temp.setY2(lhs.y2);
	temp.setG(lhs.grid); temp.setB(lhs.block);
	//Execute GPU duals:
	cR2_Mult_RHS << <lhs.grid, lhs.block >> >(temp.getD_ptr(), lhs.d_ptr, rhs, lhs.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 operator/(const cR2 & lhs, double rhs) {
	cR2 temp(lhs.m, lhs.n);
	if (rhs != 0.0) {
		temp.setDX(lhs.dx); temp.setDY(lhs.dy);

		temp.setX1(lhs.x1); temp.setX2(lhs.x2);
		temp.setY1(lhs.y1); temp.setY2(lhs.y2);
		temp.setG(lhs.grid); temp.setB(lhs.block);
		//Execute GPU duals:
		cR2_Divide_RHS << <lhs.grid, lhs.block >> > (temp.getD_ptr(), lhs.d_ptr, rhs, lhs.getSize());

		//Optional: COPY to host, find better place!
		//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
		return temp;
	}
	else {
		cout << "Division by zero is not allowed" << endl;
		return temp;
	}
}

cR2 operator+(double lhs, const cR2 & rhs) {
	cR2 temp(rhs.m, rhs.n);
	temp.setDX(rhs.dx); temp.setDY(rhs.dy);

	temp.setX1(rhs.x1); temp.setX2(rhs.x2);
	temp.setY1(rhs.y1); temp.setY2(rhs.y2);
	temp.setG(rhs.grid); temp.setB(rhs.block);
	//Execute GPU duals:
	cR2_Add_LHS << <rhs.grid, rhs.block >> >(temp.getD_ptr(), lhs, rhs.d_ptr, rhs.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 operator-(double lhs, const cR2 & rhs) {
	cR2 temp(rhs.m, rhs.n);
	temp.setDX(rhs.dx); temp.setDY(rhs.dy);

	temp.setX1(rhs.x1); temp.setX2(rhs.x2);
	temp.setY1(rhs.y1); temp.setY2(rhs.y2);
	temp.setG(rhs.grid); temp.setB(rhs.block);
	//Execute GPU duals:
	cR2_Sub_LHS << <rhs.grid, rhs.block >> >(temp.getD_ptr(), lhs, rhs.d_ptr, rhs.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 operator*(double lhs, const cR2 & rhs) {
	cR2 temp(rhs.m, rhs.n);
	temp.setDX(rhs.dx); temp.setDY(rhs.dy);

	temp.setX1(rhs.x1); temp.setX2(rhs.x2);
	temp.setY1(rhs.y1); temp.setY2(rhs.y2);
	temp.setG(rhs.grid); temp.setB(rhs.block);
	//Execute GPU duals:
	cR2_Mult_LHS << <rhs.grid, rhs.block >> >(temp.getD_ptr(), lhs, rhs.d_ptr, rhs.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 operator/(double lhs, const cR2 & rhs) {
	cR2 temp(rhs.m, rhs.n);
	temp.setDX(rhs.dx); temp.setDY(rhs.dy);

	temp.setX1(rhs.x1); temp.setX2(rhs.x2);
	temp.setY1(rhs.y1); temp.setY2(rhs.y2);
	temp.setG(rhs.grid); temp.setB(rhs.block);
	//Execute GPU duals:
	cR2_Divide_LHS << <rhs.grid, rhs.block >> >(temp.getD_ptr(), lhs, rhs.d_ptr, rhs.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}


cR2 d_dx(const cR2 & f, int Ver) {
	cR2 temp(f);
	temp.call2Host();
	int M = f.m, N = f.n;
	if (Ver == 1) {
		for (int j = 0; j < N; ++j) {
			for (int i = 0; i < M - 1; ++i) {
				temp.h_ptr[i + M*j] = (f.h_ptr[i + 1 + M*j] - f.h_ptr[i + M*j]) / (f.dx);
			}
			temp.h_ptr[M - 1 + M*j] = 0.75*temp.h_ptr[M - 1 + M*j];
		}
		temp.send2Device();
		return temp;
	}
	else if (Ver == 2) {
		for (int j = 0; j < N; ++j) {
			for (int i = 1; i < M - 1; ++i) {
				temp.h_ptr[i + M*j] = (f.h_ptr[i + 1 + M*j] - f.h_ptr[i - 1 + M*j]) / (2 * f.dx);
			}
			temp.h_ptr[M*j] = 0.75*temp.h_ptr[1 + M*j];
			temp.h_ptr[M - 1 + M*j] = 0.75*temp.h_ptr[M - 2 + M*j];
		}
		temp.send2Device();
		return temp;
	}
	else if (Ver == 3) {
		for (int j = 0; j < N; ++j) {
			for (int i = 2; i < M - 2; ++i) {
				temp.h_ptr[i + M*j] = 2 * (f.h_ptr[i + 1 + M*j] - f.h_ptr[i - 1 + M*j]) / (3 * f.dx)
					- (f.h_ptr[i + 2 + M*j] - f.h_ptr[i - 2 + M*j]) / (12 * f.dx);
			}
			temp.h_ptr[1 + M*j] = 0.75*temp.h_ptr[2 + M*j];
			temp.h_ptr[M*j] = 0.75*temp.h_ptr[1 + M*j];
			temp.h_ptr[M - 2 + M*j] = 0.75*temp.h_ptr[M - 3 + M*j];
			temp.h_ptr[M - 1 + M*j] = 0.75*temp.h_ptr[M - 2 + M*j];
		}
		temp.send2Device();
		return temp;
	}
	else {
		cout << "Invalid option" << endl;
		return temp;
	}
}

cR2 d_dy(const cR2 & f, int Ver) {
	cR2 temp(f);
	temp.call2Host();
	int M = f.m, N = f.n;
	if (Ver == 1) {
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N - 1; ++j) {
				temp.h_ptr[i + M*j] = (f.h_ptr[i + M*(j + 1)] - f.h_ptr[i + M*j]) / (f.dy);
			}
			temp.h_ptr[i] = 0.75*temp.h_ptr[i + M];
		}
		temp.send2Device();
		return temp;
	}
	else if (Ver == 2) {
		for (int i = 0; i < M; ++i) {
			for (int j = 1; j < N - 1; ++j) {
				temp.h_ptr[i + M*j] = (f.h_ptr[i + M*(j + 1)] - f.h_ptr[i + M*(j - 1)]) / (2 * f.dx);
			}
			temp.h_ptr[i] = 0.75*temp.h_ptr[i + M];
			temp.h_ptr[i + M*(N - 1)] = 0.75*temp.h_ptr[i + M*(N - 2)];
		}
		temp.send2Device();
		return temp;
	}
	else if (Ver == 3) {
		for (int i = 0; i < N; ++i) {
			for (int j = 2; j < N - 2; ++j) {
				temp.h_ptr[i + M*j] = 2 * (f.h_ptr[i + M*(j + 1)] - f.h_ptr[i + M*(j - 1)]) / (3 * f.dx)
					- (f.h_ptr[i + M*(j + 2)] - f.h_ptr[i + M*(j - 2)]) / (12 * f.dx);
			}
			temp.h_ptr[i + M] = 0.75*temp.h_ptr[i + M * 2];
			temp.h_ptr[i] = 0.75*temp.h_ptr[i + M];
			temp.h_ptr[i + M*(N - 2)] = 0.75*temp.h_ptr[i + M*(N - 3)];
			temp.h_ptr[i + M*(N - 1)] = 0.75*temp.h_ptr[i + M*(N - 2)];
		}
		temp.send2Device();
		return temp;
	}
	else {
		cout << "Invalid option" << endl;
		return temp;
	}
}

cR2 normalize(cR2 & f) {
	//Normalizes R2 function.
	cR2 temp(f);
	temp.call2Host();
	double Max = abs(f.getMax());
	if (Max == 0) {
		temp = f;
	}
	else {
		temp = f / Max;
	}
	return temp;
}
//Wurks
double cR2::getMax() const {
	//Call to host first!
	//double temp = max_element<double>(h_ptr, h_ptr + m*n);
	double temp = 0.0;
	for (int i = 0; i < m*n; ++i) {
		if (abs(h_ptr[i]) > abs(temp)) {
			temp = h_ptr[i];
		}
	}//*/
	return temp;
}

void cR2::send2Device() {
	//Use to copy data from device to host.
	//Use minimally as possible.
	cudaMemcpy(d_ptr, h_ptr, m*n * sizeof(double), cudaMemcpyHostToDevice);
}

void cR2::call2Host() {
	//Use to copy data from device to host.
	//Use minimally as possible.
	cudaMemcpy(h_ptr, d_ptr, m*n * sizeof(double), cudaMemcpyDeviceToHost);
}

bool sameDim(const cR2 & f, const cR2 & g) {
	if ((f.x1 == g.x1) && (f.x2 == g.x2) && (f.m == g.m) && (f.y1 == g.y1) && (f.y2 == g.y2) && (f.n == g.n)) {
		return true;
	}
	else {
		return false;
	}
}

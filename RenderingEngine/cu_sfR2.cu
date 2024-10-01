#include "cu_sfR2.h"

#include <cuda.h>
#include <math_constants.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gsl/gsl_poly.h"

#include "stdud.h"
#include "VecR2.h"
using namespace std;
/*
__global__ void Gen4Vec(float* camPos, float** camDir, double* rotT, double* rotP,
	double* t, double* r, double* theta, double* phi) {
	int
		x = threadIdx.x + blockIdx.x * blockDim.x;
	int
		y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	//Phi rotation.
	if (y == 255) {
		r[offset] = camDir[0][1] * cos(rotP[x]) + (camDir[2][2] * camDir[0][3] - camDir[2][3] * camDir[0][2])*sin(rotP[x]);
		theta[offset] = camDir[0][2] * cos(rotP[x]) + (camDir[2][3] * camDir[0][1] - camDir[2][1] * camDir[0][3])*sin(rotP[x]);
		phi[offset] = camDir[0][2] * cos(rotP[x]) + (camDir[2][1] * camDir[0][2] - camDir[2][2] * camDir[0][1])*sin(rotP[x]);
	}

	//Theta rotation.
	if (x == 255) {
		r[offset] = camDir[0][1] * cos(rotP[x]) + (camDir[1][2] * camDir[0][3] - camDir[1][3] * camDir[0][2])*sin(rotP[x]);
		theta[offset] = camDir[0][2] * cos(rotP[x]) + (camDir[1][3] * camDir[0][1] - camDir[1][1] * camDir[0][3])*sin(rotP[x]);
		phi[offset] = camDir[0][2] * cos(rotP[x]) + (camDir[1][1] * camDir[0][2] - camDir[1][2] * camDir[0][1])*sin(rotP[x]);
	}
	//Rotate around up
}

__global__ void Compose(float* camPos, float** camDir, double* rotT, double* rotP,
	double* t, double* r, double* theta, double* phi) {
	int
		x = threadIdx.x + blockIdx.x * blockDim.x;
	int
		y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (x != y) {
		r[offset] = r[255 + y * blockDim.x * gridDim.x] + r[x + 255 * blockDim.x * gridDim.x];
		theta[offset] = theta[255 + y * blockDim.x * gridDim.x] + theta[x + 255 * blockDim.x * gridDim.x];
		phi[offset] = phi[255 + y * blockDim.x * gridDim.x] + phi[x + 255 * blockDim.x * gridDim.x];
	}
}
//*/
__global__ void cR2_SIN(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = sin(a[i]);
	}
}

__global__ void cR2_COS(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = cos(a[i]);
	}
}

__global__ void cR2_TAN(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		if (cos(a[i]) != 0.0) {
			b[i] = sin(a[i]) / cos(a[i]);
		}
		else {
			b[i] = 0.0;
		}
	}
}


__global__ void cR2_SINH(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = sinh(a[i]);
	}
}

__global__ void cR2_COSH(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = cosh(a[i]);
	}
}

__global__ void cR2_TANH(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = sinh(a[i]) / cosh(a[i]);
	}
}

__global__ void cR2_SQRT(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = sqrt(a[i]);
	}
}

__global__ void cR2_EXP(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = exp(a[i]);
	}
}

__global__ void cR2_POW(double* c, double* a, double b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = pow(a[i], b);
	}
}

__global__ void cR2_POW(double* c, double a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = pow(a, b[i]);
	}
}

__global__ void cR2_POW(double* c, double* a, double* b, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		c[i] = pow(a[i],b[i]);
	}
}

__global__ void cR2_ASIN(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		if (abs(a[i]) < 1.0) {
			b[i] = asin(a[i]);
		}
		else {
			b[i] = 0.0;
		}
	}
}

__global__ void cR2_ACOS(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		if (abs(a[i]) < 1.0) {
			b[i] = acos(a[i]);
		}
		else {
			b[i] = 0.0;
		}
	}
}

__global__ void cR2_ATAN(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = atan(a[i]);
	}
}

__global__ void cR2_ATAN(double* b, double* y, double* x, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = atan2(y[i], x[i]);
	}
}

__global__ void cR2_ERF(double* b, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = erf(a[i]);
	}
}

__global__ void cR2_J_n(double* b, int n, double* a, int N) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		b[i] = jn(n, a[i]);
	}
}

__global__ void S_temporal(double *out_ptr, double *arg, int size, double Rs) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size) {
		/*
		if (arg[i]>Rs) {
			double h = sqrt(1 - Rs / arg[i]);
			out_ptr[i] = arg[i] * h + (Rs / 2)*log(abs((1 + h) / (1 - h)));
		}
		else {
			out_ptr[i] = CUDART_NAN;
		}//*/
		double h = sqrt(1 - Rs / arg[i]);
		out_ptr[i] = arg[i] * h + (Rs / 2)*log(abs((1 + h) / (1 - h)));
	}
}
/*
void Gen4Vec(float * d_camPos, float ** d_camDir, cR1 & rotTheta, cR1 & rotPhi, cR2 & tDot, cR2 & rDot, cR2 & thetaDot, cR2 & phiDot) {
	dim3 grid(512 / 32, 512 / 32);
	dim3 block(32, 32);
	Gen4Vec << <grid, block >> > (d_camPos, d_camDir, rotTheta.getD_ptr(), rotPhi.getD_ptr(),
		tDot.getD_ptr(), rDot.getD_ptr(), thetaDot.getD_ptr(), phiDot.getD_ptr());
}
//*/
cR2 sin(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_SIN << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 cos(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_COS << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 tan(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_TAN << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 sinh(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_SINH << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 cosh(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_COSH << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 tanh(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_TANH << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 sqrt(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_SQRT << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 exp(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_EXP << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 pow(const cR2 & base, double exp) {
	cR2 temp(base.getIndex1(), base.getIndex2());
	temp.setDX(base.getDX()); temp.setDY(base.getDY());
	temp.setX1(base.getX1()); temp.setX2(base.getX2());
	temp.setY1(base.getY1()); temp.setY2(base.getY2());
	temp.setG(base.getG()); temp.setB(base.getB());
	//Execute GPU duals:
	cR2_POW << <base.getG(), base.getB() >> > (temp.getD_ptr(), base.getD_ptr(), exp, base.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 pow(double base, const cR2 & exp) {
	cR2 temp(exp.getIndex1(), exp.getIndex2());
	temp.setDX(exp.getDX()); temp.setDY(exp.getDY());
	temp.setX1(exp.getX1()); temp.setX2(exp.getX2());
	temp.setY1(exp.getY1()); temp.setY2(exp.getY2());
	temp.setG(exp.getG()); temp.setB(exp.getB());
	//Execute GPU duals:
	cR2_POW << <exp.getG(), exp.getB() >> > (temp.getD_ptr(), base, exp.getD_ptr(), exp.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 pow(const cR2 & base, const cR2 & exp) {
	cR2 temp(base.getIndex1(), base.getIndex2());
	temp.setDX(base.getDX()); temp.setDY(base.getDY());
	temp.setX1(base.getX1()); temp.setX2(base.getX2());
	temp.setY1(base.getY1()); temp.setY2(base.getY2());
	temp.setG(base.getG()); temp.setB(base.getB());
	//Execute GPU duals:
	cR2_POW << <base.getG(), base.getB() >> > (temp.getD_ptr(), base.getD_ptr(), exp.getD_ptr(), base.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 asin(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_ASIN << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 acos(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_ACOS << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 atan(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_ATAN << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 atan(const cR2 & y, const cR2 & x) {
	cR2 temp(x.getIndex1(), x.getIndex2());
	temp.setDX(x.getDX()); temp.setDY(x.getDY());
	temp.setX1(x.getX1()); temp.setX2(x.getX2());
	temp.setY1(x.getY1()); temp.setY2(x.getY2());
	temp.setG(x.getG()); temp.setB(x.getB());
	//Execute GPU duals:
	cR2_ATAN << <x.getG(), x.getB() >> > (temp.getD_ptr(), y.getD_ptr(), x.getD_ptr(), x.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 erf(const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_ERF << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 jn(int n, const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_J_n << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), n, arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 ln(const cR2 & arg) {
	//Fix this shit!
	cR2 temp(arg.getIndex1(), arg.getIndex2());
	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());
	//Execute GPU duals:
	cR2_ERF << <arg.getG(), arg.getB() >> > (temp.getD_ptr(), arg.getD_ptr(), arg.getSize());

	//Optional: COPY to host, find better place!
	//cudaMemcpy(h_yy, d_yy, m*n * sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}

cR2 S_temporal(double Rs, const cR2 & arg) {
	cR2 temp(arg.getIndex1(), arg.getIndex2());

	temp.setDX(arg.getDX()); temp.setDY(arg.getDY());
	temp.setX1(arg.getX1()); temp.setX2(arg.getX2());
	temp.setY1(arg.getY1()); temp.setY2(arg.getY2());
	temp.setG(arg.getG()); temp.setB(arg.getB());

	S_temporal<<<arg.getG(), arg.getB()>>>(temp.getD_ptr(), arg.getD_ptr(), arg.getSize(), Rs);

	return temp;
}
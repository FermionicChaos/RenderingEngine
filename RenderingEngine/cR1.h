#ifndef CR1_H
#define CR1_H
#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class cR1 {
private:
	double *h_ptr, *d_ptr;
	int m; // Number of elements in Array
	double dx;
	double x1, x2;
	dim3 grid, block;
public:
	cR1(); //Default Constructor (No Reserve)
	~cR1(); //Destructor
	cR1(const cR1& inp); // Copy Constructor R1 yy = xx;
	cR1& operator=(const cR1& rhs); // yy = xx
	cR1(cR1&& inp);
	cR1& operator=(cR1&& rhs);
	cR1(int size); //Reserve space

	cR1(double a, double b, int size);

	double cR1::operator()(int I);
	double cR1::operator()(double p);

	cR1 operator+(const cR1& rhs);
	cR1 operator-(const cR1& rhs);
	cR1 operator*(const cR1& rhs);
	cR1 operator/(const cR1& rhs);

	friend cR1 operator+(const cR1& lhs, double rhs);
	friend cR1 operator-(const cR1& lhs, double rhs);
	friend cR1 operator*(const cR1& lhs, double rhs);
	friend cR1 operator/(const cR1& lhs, double rhs);

	friend cR1 operator+(double lhs, const cR1& rhs);
	friend cR1 operator-(double lhs, const cR1& rhs);
	friend cR1 operator*(double lhs, const cR1& rhs);
	friend cR1 operator/(double lhs, const cR1& rhs);

	friend cR1 d_dx(const cR1& f, int Ver);
	friend double integralD(double a, double b, const cR1& f);

	int getIndex(double p);
	double* getH_ptr() const { return h_ptr; }
	double* getD_ptr() const { return d_ptr; }
	int getSize() const { return m; }
	double getDX() const { return dx; }
	double getX1() const { return x1; }
	double getX2() const { return x2; }
	int getMem() const { return sizeof(double)*m; }
	dim3 getG() const { return grid; }
	dim3 getB() const { return block; }

	void setH_ptr(double* ptr) { h_ptr = ptr; }
	void setD_ptr(double* ptr) { d_ptr = ptr; }
	void setSize(int N) { m = N; }
	void setDX(double DX) { dx = DX; }
	void setX1(double X1) { x1 = X1; }
	void setX2(double X2) { x2 = X2; }
	void setG(dim3 G) { grid = G; }
	void setB(dim3 B) { block = B; }

	void send2Device();
	void call2Host();
	bool sameDim(const cR1& rhs);
};

#endif // !CR1_H

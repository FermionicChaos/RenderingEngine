#ifndef CR2_H
#define CR2_H
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdud.h"
#include "VecR2.h"
#pragma once

class cR2 {
private:
	double* h_ptr;
	double *d_ptr;
	int m, n;
	double dx, dy; // Length element 0.1 0.6 0.3 -0.1
	double x1, x2, y1, y2; //Bounds of the number plane.
	dim3 grid, block;
public:
	cR2(); //Default Constructor
	~cR2(); //Destructor
	cR2(const cR2& inp); //Copy Construct
	cR2& operator=(const cR2& rhs); //Copy Assignment
	cR2(cR2&& inp); //Move Construct
	cR2& operator=(cR2&& rhs); //Move Assignment
	cR2(int M, int N); //Reserve Memory

	cR2(int k, double X1, double X2, int M, double Y1, double Y2, int N); //Create & set domain.

	double cR2::operator()(int I, int J);
	double cR2::operator()(double X, double Y);
	double cR2::operator()(VecR2& pos);
	//Fully operational + - * /
	cR2 operator+(const cR2& rhs);
	cR2 operator-(const cR2& rhs);
	cR2 operator*(const cR2& rhs);
	cR2 operator/(const cR2& rhs);
	//Yeah, I know, poor notation. At least there is easy to read symmettry in the
	//notation.
	friend cR2 operator+(const cR2& lhs, double rhs);
	friend cR2 operator-(const cR2& lhs, double rhs);
	friend cR2 operator*(const cR2& lhs, double rhs);
	friend cR2 operator/(const cR2& lhs, double rhs);
	//Fuck these operations below!
	friend cR2 operator+(double lhs, const cR2& rhs);
	friend cR2 operator-(double lhs, const cR2& rhs);
	friend cR2 operator*(double lhs, const cR2& rhs);
	friend cR2 operator/(double lhs, const cR2& rhs);

	//These operators have NOT been configured to the GPU.
	friend cR2 d_dx(const cR2& f, int Ver);
	friend cR2 d_dy(const cR2& f, int Ver);
	friend cR2 normalize(cR2& f);

	int getIndex1() const { return m; }
	int getIndex2() const { return n; }
	double getDX() const { return dx; }
	double getDY() const { return dy; }
	double getX1() const { return x1; }
	double getX2() const { return x2; }
	double getY1() const { return y1; }
	double getY2() const { return y2; }
	double* getH_ptr() const { return h_ptr; }
	double* getD_ptr() const { return d_ptr; }
	int getMEM() const { return sizeof(double)*m*n; }
	int getSize() const { return m*n; }
	double getMax() const;
	dim3 getG() const { return grid; }
	dim3 getB() const { return block; }

	void setIndex1(int M) { m = M; }
	void setIndex2(int N) { n = N; }
	void setDX(double DX) { dx = DX; }
	void setDY(double DY) { dy = DY; }
	void setX1(double X1) { x1 = X1; }
	void setX2(double X2) { x2 = X2; }
	void setY1(double Y1) { y1 = Y1; }
	void setY2(double Y2) { y2 = Y2; }
	void setH_ptr(double* B) { h_ptr = B; }
	void setD_ptr(double* B) { d_ptr = B; }
	void setG(dim3 G) { grid = G; }
	void setB(dim3 B) { block = B; }

	void send2Device();
	void call2Host();
	friend bool sameDim(const cR2& f, const cR2& g);
};


#endif // CR2_H

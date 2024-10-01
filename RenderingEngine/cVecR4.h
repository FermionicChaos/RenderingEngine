#ifndef CVECR4_H
#define CVECR4_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum metric_t {
	FLAT = 0,
	SCHWARZSCHILD = 1,
	KERR = 2
};

enum vec_t {
	POSITION = 0,
	FIELD = 1
};

class cVecR4 {
public:
	double *d_ptr;
	double *h_ptr;
	vec_t vec;
	metric_t metric;

	dim3 grid;
	dim3 block;


	cVecR4();
	~cVecR4();
	cVecR4(cVecR4& inp);
	cVecR4& operator=(cVecR4& rhs);
	cVecR4(cVecR4&& inp);
	cVecR4& operator=(cVecR4&& rhs);

	cVecR4(vec_t v_type, metric_t m_type, double z0, double z1, double z2, double z3);

	double cVecR4::operator()(int i);
	//double cVecR4::operator*(cVecR4& rhs);

	void send2Device();
	void call2Host();
};
#endif // !CVECR4_H
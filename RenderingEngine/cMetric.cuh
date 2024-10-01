#ifndef CMETRIC_CUH
#define CMETRIC_CUH

#include <cuda.h>
#include "cuda_runtime.h"
#include <math_constants.h>
#include "device_launch_parameters.h"

/*
 index (t,x,y,x) = (0,1,2,3)
*/

//#define C 3.0 * 10 ^ 8;


enum metric_t {
	FLAT = 0,
	SCHWARZSCHILD = 1,
	KERR = 2
};

__device__ float metric(metric_t type, int i, int j, float* pos, float Rs) {
	//Metric function list.
	if (type == FLAT) {
		if ((i == j) && (i == 0)) {
			return -1.0f;
		}
		else if ((i == j) && (i >= 0)) {
			return 1.0f;
		}
	}
	else if(type == SCHWARZSCHILD) {
		if ((i == j) && (i == 0)) {
			return -1.0f*(1.0f - Rs/pos[1]);
		}
		else if ((i == j) && (i == 1)) {
			return 1.0f/(1.0f - Rs / pos[1]);
		}
		else if ((i == j) && (i == 2)) {
			return pos[1] * pos[1];
		}
		else if ((i == j) && (i == 3)) {
			return pos[1] * pos[1]*sin(pos[2])*sin(pos[2]);
		}
		return sin(1.2);
	}
	else if (type = KERR) {
		return 1.0;
	}
	else {
		return 0.0;
	}
}

#endif // !CMETRIC_CUH
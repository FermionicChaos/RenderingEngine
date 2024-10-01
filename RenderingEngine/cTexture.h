#ifndef CTEXTURE_H
#define CTEXTURE_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "stb_image.h"

#include "cR1.h"
#include "cR2.h"
#include "Camera.h"
#include "Skybox.h"
typedef unsigned char uchar;
/*
The next goal is to map directly from a texture object to a cuda array. It might speed
up calculations, it might not. It is unknown at the moment and I have no way of measuring
this either. The first goal at hand is to build a curved spacetime renderer in cuda and
render the pixels to OpenGL for display. Currently I am using a GTX 1080 for CUDA calculations.
This device grants me a total of 2560 cuda cores. 

Now I need to make a UI for:

FPS
Resolution
Position & velocity


(Resume)
(Options)
(Exit)
We need to get a main Game class to keep track of everything we need before rendering, and
possibly a script extender/reader for user defined metrics.


*/

class cTexture {
public:
	uchar** h_rgba;
	uchar** d_rgba;
	char** filename;

	float* d_camPos;
	float** d_camDir;
	float* h_camPos;
	float** h_camDir;

	//Holder for final Theta and Phi.
	cR1 lambda1;
	cR1 lambda2;
	cR1 lambda3;

	cR2 THETA, PHI;
	cR2 map;

	float radius;
	double fov;
	cR1 rotTheta, rotPhi;
	cR2 tDot, rDot, thetaDot, phiDot;
	cR2 k2;

	cR2 uu, kk, ff, ff2;
	cR2 rr, theta;
	cR2 lr;
	double Rs;
	double aspectRatio;

	int *width, *height, *bpp;

	GLuint HEIGHT, WIDTH, BPP; //Parameters of pixel/texture data.
	GLuint bufferID; //OpenGL context handle for pixel data.
	GLuint textureID; //Image texture id after mapping from buffer.
	uchar4 *h_ptr; //Pixel data in host memory.
	uchar4 *d_ptr; //Pixel data in device memory.
	cudaArray_t arrayPtr; //Pixel data in device texture memory. (Unused at current moment)
	cudaGraphicsResource_t cudaResource;  //Used in mapping buffer object to cuda.
	size_t size; //Size of data in memory.

	dim3 grid, block; //Dimensional thread specifications for CUDA calculations.

	cudaDeviceProp prop;
	int dev;


	cTexture();
	~cTexture();

	void GenBuf();
	void GenTex();

	void Generate(GLuint width, GLuint height);
	void Update(double dt, Camera& camera);
	void Draw();
	void testCalibration();
	void loadImage();
	void deleteImage();
};


#endif // !CTEXTURE_H
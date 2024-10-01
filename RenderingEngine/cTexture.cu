#include "cTexture.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <cuda.h>
#include <math_constants.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaGL.h"
#include "cuda_gl_interop.h"

#include <GL/glew.h>
#include "stb_image.h"

#include "stdud.h"

#include "cR1.h"
#include "cR2.h"
#include "cu_sfR2.h"
#include "Camera.h"
#include "Skybox.h"
typedef unsigned char uchar;

void handleError(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("Error: %s\n in file %s at line no. %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}

void transform2SPH(float* h_camPos, float** h_camDir, float dt, Camera& camera) {
	//Map
	h_camPos[0] += dt;
	h_camPos[1] = sqrt(camera.camPos.x*camera.camPos.x + camera.camPos.y*camera.camPos.y + camera.camPos.z*camera.camPos.z);
	h_camPos[2] = acos(camera.camPos.z/(h_camPos[1]));
	h_camPos[3] = atan2(camera.camDir.y, camera.camDir.x);
	//Direction
	h_camDir[0][0] = 0.0f;
	h_camDir[0][1] = cos(h_camPos[3])*sin(h_camPos[2])*camera.camDir.x + sin(h_camPos[3])*sin(h_camPos[2])*camera.camDir.y + cos(h_camPos[2])*camera.camDir.z;
	h_camDir[0][2] = cos(h_camPos[3])*cos(h_camPos[2])*camera.camDir.x + sin(h_camPos[3])*cos(h_camPos[2])*camera.camDir.y - sin(h_camPos[2])*camera.camDir.z;
	h_camDir[0][3] = (-1.0f)*sin(h_camPos[3])*camera.camDir.x + cos(h_camPos[3])*camera.camDir.y;
	//Right on Screen
	h_camDir[1][0] = 0.0f;
	h_camDir[1][1] = cos(h_camPos[3])*sin(h_camPos[2])*camera.camRight.x + sin(h_camPos[3])*sin(h_camPos[2])*camera.camRight.y + cos(h_camPos[2])*camera.camRight.z;
	h_camDir[1][2] = cos(h_camPos[3])*cos(h_camPos[2])*camera.camRight.x + sin(h_camPos[3])*cos(h_camPos[2])*camera.camRight.y - sin(h_camPos[2])*camera.camRight.z;
	h_camDir[1][3] = (-1.0f)*sin(h_camPos[3])*camera.camRight.x + cos(h_camPos[3])*camera.camRight.y;
	//Up on Screen
	h_camDir[2][0] = 0.0f;
	h_camDir[2][1] = cos(h_camPos[3])*sin(h_camPos[2])*camera.camUp.x + sin(h_camPos[3])*sin(h_camPos[2])*camera.camUp.y + cos(h_camPos[2])*camera.camUp.z;
	h_camDir[2][2] = cos(h_camPos[3])*cos(h_camPos[2])*camera.camUp.x + sin(h_camPos[3])*cos(h_camPos[2])*camera.camUp.y - sin(h_camPos[2])*camera.camUp.z;
	h_camDir[2][3] = (-1.0f)*sin(h_camPos[3])*camera.camUp.x + cos(h_camPos[3])*camera.camUp.y;
	//std::system("cls");
	//printf("fps: %f\n", 1.0/dt);
	/*
	printf("acos() = %f\n", acos(1));
	printf("Position (x,y,z) = (%f,%f,%f)\n", 
		camera.camPos.x,
		camera.camPos.y,
		camera.camPos.z);

	printf("Position (r,theta,phi) = (%f,%f,%f)\n",
		h_camPos[1],
		h_camPos[2],
		h_camPos[3]
	);

	printf("Direction (r,dirTheta,dirPhi) = (%f,%f,%f)\n",
		h_camDir[0][1],
		h_camDir[0][2],
		h_camDir[0][3]
	);
	//*/
}

#define DIM 512
#define PI 3.14

extern "C" __global__ void kernel(uchar4 *ptr) {
	// map from threadIdx/BlockIdx to pixel position
	int
		x = threadIdx.x + blockIdx.x * blockDim.x;
	int
		y = threadIdx.y + blockIdx.y * blockDim.y;
	int
		offset = x + y * blockDim.x * gridDim.x;
	// now calculate the value at that position
	float
		fx = x / (float)DIM - 0.5f;
	float
		fy = y / (float)DIM - 0.5f;
	unsigned char
		green = 128 + 127 *
		sin(abs(fx * 100) - abs(fy * 100));
	// accessing uchar4 vs. unsigned char*
	ptr[offset].x = 0;		//R
	ptr[offset].y = green;	//G
	ptr[offset].z = 0;		//B
	ptr[offset].w = 255;	//A
}

__global__ void rot_PHI(float* A, float* B, double* rot, double* r, double* theta, double* phi) {
	int
		x = threadIdx.x + blockIdx.x * blockDim.x;
	int
		y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if ((x<512)&&(x<512)) {
		//Phi rotation.
		if (y == 255) {
			r[offset] = B[1] * cos(rot[x]) +(A[2] * B[3] - A[3] * B[2])*sin(rot[x]);
			theta[offset] = B[2] * cos(rot[x]) + (A[3] * B[1] - A[1] * B[3])*sin(rot[x]);
			phi[offset] = B[3] * cos(rot[x]) + (A[1] * B[2] - A[2] * B[1])*sin(rot[x]);
		}
		/*
		//Theta rotation.
		if (x == 255) {
			r[offset] = camDir[0][1] * cos(rotP[x]) + (camDir[1][2] * camDir[0][3] - camDir[1][3] * camDir[0][2])*sin(rotP[x]);
			theta[offset] = camDir[0][2] * cos(rotP[x]) + (camDir[1][3] * camDir[0][1] - camDir[1][1] * camDir[0][3])*sin(rotP[x]);
			phi[offset] = camDir[0][3] * cos(rotP[x]) + (camDir[1][1] * camDir[0][2] - camDir[1][2] * camDir[0][1])*sin(rotP[x]);
		}
		//*/
		//Rotate around up
	}
}

__global__ void rot_THETA(float* A, float* B, double* rot, double* r, double* theta, double* phi) {
	int
		x = threadIdx.x + blockIdx.x * blockDim.x;
	int
		y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if ((x<512) && (x<512)) {
		//Theta rotation.
		if (x == 255) {
			r[offset] = B[1] * cos(rot[x]) + (A[2] * B[3] - A[3] * B[2])*sin(rot[x]);
			theta[offset] = B[2] * cos(rot[x]) + (A[3] * B[1] - A[1] * B[3])*sin(rot[x]);
			phi[offset] = B[3] * cos(rot[x]) + (A[1] * B[2] - A[2] * B[1])*sin(rot[x]);
		}
	}
}
extern "C" __global__ void Compose(double* t, double* r, double* theta, double* phi) {
	int
		x = threadIdx.x + blockIdx.x * blockDim.x;
	int
		y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if ( x != y ) {
		r[offset] = r[255 + y * blockDim.x * gridDim.x] + r[x + 255 * blockDim.x * gridDim.x];
		theta[offset] = theta[255 + y * blockDim.x * gridDim.x] + theta[x + 255 * blockDim.x * gridDim.x];
		phi[offset] = phi[255 + y * blockDim.x * gridDim.x] + phi[x + 255 * blockDim.x * gridDim.x];
	}
}

extern "C" __global__ void find_TDOT(float* pos, double* tdot, double* rdot, double* thetadot, double* phidot, float Rs) {
	int
		x = threadIdx.x + blockIdx.x * blockDim.x;
	int
		y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if ((x < 512) && (x < 512)) {
		//positive tdot
		tdot[offset] = sqrt(
			rdot[offset]* rdot[offset] + 
			thetadot[offset]* thetadot[offset] * pos[1]*pos[1]/(1- Rs/pos[1]) + 
			phidot[offset]* phidot[offset] * pos[1] * pos[1]*sin(pos[2])*sin(pos[2])/ (1 - Rs / pos[1])
		);
	}
}

extern "C" __global__ void SPH(float* pos, float** dir) {
	int
		x = threadIdx.x + blockIdx.x * blockDim.x;
	int
		y = threadIdx.y + blockIdx.y * blockDim.y;
	pos[1] = sqrt(pos[1]*pos[1] + pos[2]*pos[2] + pos[3]*pos[3]);
}

extern "C" __global__ void map2Image(uchar4 *ptr, double *data, int M, int N) {
	// map from threadIdx/BlockIdx to pixel position
	int
		x = threadIdx.x + blockIdx.x * blockDim.x;
	int
		y = threadIdx.y + blockIdx.y * blockDim.y;
	int
		offset = x + y * blockDim.x * gridDim.x;
	// now calculate the value at that position
	unsigned char
		green = 255 * abs(data[offset]);
		//green = 127 * abs(fy);
		//green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));
	// accessing uchar4 vs. unsigned char*
	ptr[offset].x = green;	//R
	ptr[offset].y = green;	//G
	ptr[offset].z = green;	//B
	ptr[offset].w = 255;	//A
}

extern "C" __global__ void imageLoad(uchar4* out_rgba, uchar* in_rgba) {
	// map from threadIdx/BlockIdx to pixel position
	int
		x = threadIdx.x + blockIdx.x * blockDim.x;
	int
		y = threadIdx.y + blockIdx.y * blockDim.y;
	int
		offset = x + y * blockDim.x * gridDim.x;
	out_rgba[offset].x = in_rgba[0 + 4*offset];
	out_rgba[offset].y = in_rgba[1 + 4 * offset];
	out_rgba[offset].z = in_rgba[2 + 4 * offset];
	out_rgba[offset].w = in_rgba[3 + 4 * offset];
}

cTexture::cTexture() {
	radius = 1.0f;
	fov = PI / 6.0;
	h_camPos = new float[4];
	h_camDir = new float*[3];
	cudaMalloc(&d_camPos, 4*sizeof(float));
	d_camDir = new float*[3];

	for (int i = 0; i < 3; i++) {
		h_camDir[i] = new float[4];
		cudaMalloc(&d_camDir[i], 4 * sizeof(float));
	}
}

cTexture::~cTexture() {
	//cudaGraphicsUnregisterResource(cudaResource);
	glDeleteBuffers(1, &bufferID);
	glDeleteTextures(1, &textureID);

	delete[] h_camPos;
	cudaFree(&d_camPos);
	for (int i = 0; i < 3; i++) {
		delete[] h_camDir[i];
		cudaFree(&d_camDir[i]);
	}
	//delete[] h_camDir;
	//delete[] d_camDir;
}

void cTexture::GenBuf() {
	glGenBuffers(1, &bufferID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WIDTH*HEIGHT * 4, NULL, GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

void cTexture::GenTex() {
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void cTexture::Generate(GLuint width, GLuint height) {
	aspectRatio = (double)width / (double)height;
	HEIGHT = 512;
	WIDTH = 512;

	grid = dim3(WIDTH / 32, HEIGHT / 32);
	block = dim3(32, 32);

	//Generate domain for fov
	rotTheta = cR1(-fov, fov, 512);
	rotPhi = cR1(-fov*aspectRatio, fov*aspectRatio, 512);

	//reserve memory for light 4 vectors.
	tDot = cR2(512, 512);
	rDot = cR2(512, 512);
	thetaDot = cR2(512, 512);
	phiDot = cR2(512, 512);

	///*
	double X1 = 0.0;
	double X2 = 1.0;
	double Y1 = 0.0;
	double Y2 = 100.0;

	uu = cR2(1, X1, X2, 1024, Y1, Y2, 1024); //Create & set domain.
	kk = cR2(2, X1, X2, 1024, Y1, Y2, 1024); //Create & set domain.
	//dalpha = 0.0f;
	//ff = 1.0 / (sqrt(uu*uu*uu - uu*uu + kk*kk));
	//*/

	glGenBuffers(1, &bufferID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WIDTH*HEIGHT * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

}

void cTexture::Update(double dt, Camera& camera) {
	///*
	//Setup Cam vectors
	transform2SPH(h_camPos, h_camDir, dt, camera);
	cudaMemcpy(d_camPos, h_camPos, 4 * sizeof(float), cudaMemcpyHostToDevice);
	for (int i = 0; i < 3; i++) {
		cudaMemcpy(d_camDir[i], h_camDir[i], 4 * sizeof(float), cudaMemcpyHostToDevice);
	}
	///*
	//Setup lightpath vectors
	rot_PHI << < grid, block >> > (d_camDir[1], d_camDir[0], rotPhi.getD_ptr(), rDot.getD_ptr(), thetaDot.getD_ptr(), phiDot.getD_ptr());
	rot_THETA << < grid, block >> > (d_camDir[2], d_camDir[0], rotTheta.getD_ptr(), rDot.getD_ptr(), thetaDot.getD_ptr(), phiDot.getD_ptr());
	//VERIFY!
	Compose << <grid, block >> > (tDot.getD_ptr(), rDot.getD_ptr(), thetaDot.getD_ptr(), phiDot.getD_ptr());
	find_TDOT << <grid, block >> > (d_camPos, tDot.getD_ptr(), rDot.getD_ptr(), thetaDot.getD_ptr(), phiDot.getD_ptr(), Rs);

	//Test1 r<1.5 Rs 0.2 rDot^2> V?
	//test2 
	//*/

	///*
	cudaGraphicsGLRegisterBuffer(&cudaResource, bufferID, cudaGraphicsMapFlagsNone);
	cudaGraphicsMapResources(1, &cudaResource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&d_ptr, &size, cudaResource);

	imageLoad << <grid, block >> > (d_ptr, d_rgba[0]);

	cudaGraphicsUnmapResources(1, &cudaResource, NULL);
	cudaGraphicsUnregisterResource(cudaResource);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	//*/
}

void cTexture::Draw() {
	/*
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferID);

	glBindTexture(GL_TEXTURE_2D, imageTex);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferID);
	glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glDisable(GL_DEPTH_TEST);
	glRasterPos2i(0, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	//*/
}

void cTexture::testCalibration() {
	/*
	//Calibration code.
	//Topleft to bottom right.
	int i, j, offset;
	//uchar* rgba = stbi_load("images/skybox/calibrate.png", &width, &height, &bpp, 4);
	uchar* rgba = stbi_load("images/calibration/color.jpg", &width, &height, &bpp, 4);
	printf("calibration of image importing. Correct color: BLUE\n");
	printf("i = %d, j = %d\n", i = 50, j = 50);
	offset = 4 * (i + j*width);
	printf("val.r = %X\n", rgba[0 + offset]);
	printf("val.g = %X\n", rgba[1 + offset]);
	printf("val.b = %X\n", rgba[2 + offset]);
	printf("val.a = %X\n", rgba[3 + offset]);

	printf("calibration of image importing. Correct color: GREEN\n");
	printf("i = %d, j = %d\n", i = 150, j = 50);
	offset = 4 * (i + j*width);
	printf("val.r = %X\n", rgba[0 + offset]);
	printf("val.g = %X\n", rgba[1 + offset]);
	printf("val.b = %X\n", rgba[2 + offset]);
	printf("val.a = %X\n", rgba[3 + offset]);

	printf("calibration of image importing. Correct color: RED\n");
	printf("i = %d, j = %d\n", i = 150, j = 150);
	offset = 4 * (i + j*width);
	printf("val.r = %X\n", rgba[0 + offset]);
	printf("val.g = %X\n", rgba[1 + offset]);
	printf("val.b = %X\n", rgba[2 + offset]);
	printf("val.a = %X\n", rgba[3 + offset]);

	printf("calibration of image importing. Correct color: BLACK\n");
	printf("i = %d, j = %d\n", i = 50, j = 150);
	offset = 4 * (i + j*width);
	printf("val.r = %X\n", rgba[0 + offset]);
	printf("val.g = %X\n", rgba[1 + offset]);
	printf("val.b = %X\n", rgba[2 + offset]);
	printf("val.a = %X\n", rgba[3 + offset]);

	stbi_image_free(rgba);
	//*/
}

void cTexture::loadImage() {
	filename = new char*[6];
	h_rgba = new uchar*[6];
	d_rgba = new uchar*[6];
	width = new int[6];
	height = new int[6];
	bpp = new int[6];

	for (int i = 0; i < 6; i++) {
		filename[i] = new char[256];
		strcpy(filename[i], "images/skybox/space/lightblue/");
		//printf("filename[%d] = %s\n", i, filename[i]);
	}
	strcat(filename[0], "front.png");
	strcat(filename[1], "back.png");
	strcat(filename[2], "left.png");
	strcat(filename[3], "right.png");
	strcat(filename[4], "top.png");
	strcat(filename[5], "bot.png");

	for (int i = 0; i < 6; i++) {
		printf("filename[%d] = %s\n", i, filename[i]);
		h_rgba[i] = stbi_load(filename[i], &width[i], &height[i], &bpp[i], 4);
		cudaMalloc(&d_rgba[i], width[i] * height[i] * 4);
		cudaMemcpy(d_rgba[i], h_rgba[i], 4*width[i]*height[i], cudaMemcpyHostToDevice);
	}
}

void cTexture::deleteImage() {
	for (int i = 0; i < 6; i++) {
		stbi_image_free(h_rgba[i]);
		cudaFree(d_rgba[i]);
		delete[] filename[i];
	}

	delete[] h_rgba;
	delete[] d_rgba;
	delete[] filename;
	delete[] width;
	delete[] height;
	delete[] bpp;
}

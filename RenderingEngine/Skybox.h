#ifndef SKYBOX_H
#define SKYBOX_H
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Texture.h"

class Skybox {
public:
	Texture front;
	Texture back;
	Texture left;
	Texture right;
	Texture top;
	Texture bot;

	Skybox();
	~Skybox();

	Skybox(char* PATH);

	void loadDEFAULT();
	void load(char* PATH);

};

#endif // !SKYBOX_H
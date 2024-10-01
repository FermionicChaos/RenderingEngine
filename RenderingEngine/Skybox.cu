#include "Skybox.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <GL/glew.h>
#include "stb_image.h"

#include "Texture.h"

Skybox::Skybox() {

}

Skybox::~Skybox() {
}

Skybox::Skybox(char * PATH) {
	//Load custom textures into CUDA for kernel access.
}

void Skybox::loadDEFAULT() {
	char** filename = new char*[6];
	for (int i = 0; i < 6; i++) {
		filename[i] = new char[256];
		strcpy(filename[i], "images/skybox/space/lightblue/");
	}
	strcat(filename[0], "front.png");
	strcat(filename[1], "back.png");
	strcat(filename[2], "left.png");
	strcat(filename[3], "right.png");
	strcat(filename[4], "top.png");
	strcat(filename[5], "bot.png");

	front = Texture(filename[0]);
	back = Texture(filename[1]);
	left = Texture(filename[2]);
	right = Texture(filename[3]);
	top = Texture(filename[4]);
	bot = Texture(filename[5]);
}

void Skybox::load(char * PATH) {
	char** filename = new char*[6];
	for (int i = 0; i < 6; i++) {
		filename[i] = new char[256];
		strcpy(filename[i], PATH);
	}
	strcat(filename[0], "front.png");
	strcat(filename[1], "back.png");
	strcat(filename[2], "left.png");
	strcat(filename[3], "right.png");
	strcat(filename[4], "top.png");
	strcat(filename[5], "bot.png");

	front = Texture(filename[0]);
	back = Texture(filename[1]);
	left = Texture(filename[2]);
	right = Texture(filename[3]);
	top = Texture(filename[4]);
	bot = Texture(filename[5]);
}

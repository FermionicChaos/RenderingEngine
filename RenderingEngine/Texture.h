#ifndef TEXTURE_H
#define TEXTURE_H
#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <GL/glew.h>
#include "stb_image.h"

class Texture {
public:
	GLuint bufferID;
	GLuint textureID;
	int width, height, bpp;
	char* file;

	Texture();
	Texture(char* Name);

	void load(char* Name);
};

#endif // !TEXTURE_H
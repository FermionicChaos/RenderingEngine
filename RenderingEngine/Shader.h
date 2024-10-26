#ifndef SHADER_H
#define SHADER_H
#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <GL/glew.h>

class Shader
{
public:
	GLuint Program;
	// Constructor generates the shader on the fly
	Shader();
	Shader(const GLchar* vertexPath, const GLchar* fragmentPath);
	// Uses the current shader
	void Use() { glUseProgram(this->Program); }
};

#endif // !SHADER_H
#ifndef COORDINATEAXIS_H
#define COORDINATEAXIS_H
#pragma once

// GLEW
//#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Other includes
#include "Shader.h"
#include "Texture.h"
#include "Camera.h"
using namespace std;
using namespace glm;

class CoordinateAxis {
public:
	Shader shader;
	GLuint VBO, VAO; //Order for somefucking reason matters, do not type "GLuint VAO, VBO" It will not work.
	
	vec3 posVec;
	mat4 view, position, projection;
	GLint d_view, d_position, d_projection;

	CoordinateAxis();
	~CoordinateAxis();

	void Draw(Camera &camera);
};
#endif // !COORDINATEAXIS_H

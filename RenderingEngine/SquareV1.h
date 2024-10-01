#ifndef SQUAREV1_H
#define SQUAREV1_H
#pragma once

// GLEW
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

class Square {
public:
	Shader shader;
	Texture texture1;
	Texture texture2;
	GLuint VBO, VAO; //Order for somefucking reason matters, do not type "GLuint VAO, VBO" It will not work.
	
	vec3 posVec;
	mat4 view, position, projection; //For some cocksucking reason, I have to declare mat4 everytime in this trash.
	GLint d_view, d_position, d_projection;

	GLfloat angle;

	Square();
	~Square();

	void Draw(Camera& camera);

};
#endif // !SQUAREV1_H
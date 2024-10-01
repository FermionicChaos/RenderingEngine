#include <iostream>

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

#include "CoordinateAxis.h"
using namespace std;
using namespace glm;


CoordinateAxis::CoordinateAxis() {
	//Declaration
	GLfloat vertices[] = {
		0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
		1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,

		0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,

		0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
		0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f
	};

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
	this->shader = Shader("shaders/C_axis.vsh", "shaders/C_axis.fsh");
	posVec = vec3(0.0f, 0.0f, 0.0f);
}

CoordinateAxis::~CoordinateAxis() {
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
}

void CoordinateAxis::Draw(Camera & camera) {
	//Draw to window from camera.
	shader.Use();

	// Create camera transformation
	position = glm::translate(mat4(), posVec);
	view = camera.GetViewMatrix();
	projection = glm::perspective(camera.Zoom, (float)camera.WIDTH / (float)camera.HEIGHT, 0.1f, 1000.0f);
	// Get the uniform locations in compiled shader.
	d_position = glGetUniformLocation(shader.Program, "posVec");
	d_view = glGetUniformLocation(shader.Program, "view");
	d_projection = glGetUniformLocation(shader.Program, "projection");

	//Data bound and ready to be implemented with program.
	glBindVertexArray(VAO);

	// Pass the matrices to the shader.
	glUniformMatrix4fv(d_view, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(d_projection, 1, GL_FALSE, glm::value_ptr(projection));
	glUniformMatrix4fv(d_position, 1, GL_FALSE, glm::value_ptr(position));

	glDrawArrays(GL_LINES, 0, 6);
	//Unbind VAO
	glBindVertexArray(0);
}

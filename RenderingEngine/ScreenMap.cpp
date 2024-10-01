#include "ScreenMap.h"

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

ScreenMap::ScreenMap() {
	GLfloat vertices[] = {
		1.0f,  1.0f, 0.0f,  // Top Right
		1.0f, -1.0f, 0.0f,  // Bottom Right
		-1.0f, -1.0f, 0.0f,  // Bottom Left
		-1.0f,  1.0f, 0.0f   // Top Left 
	};
	GLuint indices[] = {  // Note that we start from 0!
		0, 1, 3,  // First Triangle
		1, 2, 3   // Second Triangle
	};
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0); // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind

	glBindVertexArray(0); // Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO

	shader = Shader("shaders/OverLay.vsh", "shaders/OverLay.fsh");

	//GRID.Generate(HEIGHT, WIDTH);
	GRID.Generate(512, 512);
	//Load Texture
	texture.textureID = GRID.textureID;
}

ScreenMap::~ScreenMap() {
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
	GRID.deleteImage();
}

ScreenMap::ScreenMap(GLuint width, GLuint height) {
	WIDTH = width;
	HEIGHT = height;

	GLfloat vertices[] = {
		1.0f,  1.0f, 0.0f,  // Top Right
		1.0f, -1.0f, 0.0f,  // Bottom Right
		-1.0f, -1.0f, 0.0f,  // Bottom Left
		-1.0f,  1.0f, 0.0f   // Top Left 
	};
	GLuint indices[] = {  // Note that we start from 0!
		0, 1, 3,  // First Triangle
		1, 2, 3   // Second Triangle
	};
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0); // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind

	glBindVertexArray(0); // Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO

	shader = Shader("shaders/OverLay.vsh", "shaders/OverLay.fsh");

	//GRID.Generate(HEIGHT, WIDTH);
	GRID.Generate(width, height);
	GRID.loadImage();
}

void ScreenMap::Update(double dt, Camera& camera) {
	//Update screen!

	GRID.Update(dt, camera);
	texture.textureID = GRID.textureID;
}

void ScreenMap::Draw(Camera & camera) {
	shader.Use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture.textureID);
	glUniform1i(glGetUniformLocation(shader.Program, "ourTexture1"), 0);


	glBindVertexArray(VAO);
	//glDrawArrays(GL_TRIANGLES, 0, 6);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}



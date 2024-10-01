#include "Quad.h"
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
#include "cTexture.h"
#include "Camera.h"

Quad::Quad() {
	GLfloat V[] = {
		1.0f,  1.0f, 1.0f, 1.0f,  // Top Right
		1.0f, -1.0f, 1.0f, 0.0f,  // Bottom Right
		-1.0f, -1.0f, 0.0f, 0.0f, // Bottom Left
		-1.0f,  1.0f, 0.0f, 1.0f   // Top Left 
	};
	GLuint I[] = {  // Note that we start from 0!
		0, 1, 3,  // First Triangle
		1, 2, 3   // Second Triangle
	};

	for (int i = 0; i < 12; i++) {
		vertices[i] = V[i];
	}

	for (int i = 0; i < 6; i++) {
		indices[i] = I[i];
	}

	shader = Shader("shaders/plot.vsh", "shaders/plot.fsh");
	texture = Texture("images/wall.jpg");
}

Quad::~Quad() {

}

Quad::Quad(GLuint width, GLuint height) {
	resolution.x = width; resolution.y = height;
}

void Quad::Initialize() {
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0); // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind

	glBindVertexArray(0); // Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO
}

void Quad::Clear() {
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
}

void Quad::Update(double dt, Camera & camera) {

}

void Quad::Draw(Camera & camera) {
	shader.Use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture.textureID);
	glUniform1i(glGetUniformLocation(shader.Program, "TextureUnit0"), 0);
	glUniform2fv(glGetUniformLocation(shader.Program, "resolution"), 1, glm::value_ptr(resolution));

	glBindVertexArray(VAO);
	//glDrawArrays(GL_TRIANGLES, 0, 6);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

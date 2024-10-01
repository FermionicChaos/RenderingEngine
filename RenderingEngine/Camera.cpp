// Std. Includes
#include <vector>

// GL Includes
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Camera.h"
using namespace glm;

glm::mat4 Camera::GetViewMatrix() {
	return glm::lookAt(this->camPos, this->camPos + this->camDir, this->camUp);
}

void Camera::ProcessKeyboard(Camera_Movement direction, GLfloat deltaTime) {
	GLfloat velocity = this->MovementSpeed * deltaTime;
	if (direction == FORWARD)
		this->camPos += this->camDir * velocity;
	if (direction == BACKWARD)
		this->camPos -= this->camDir * velocity;
	if (direction == LEFT)
		this->camPos -= this->camRight * velocity;
	if (direction == RIGHT)
		this->camPos += this->camRight * velocity;
}

void Camera::ProcessMouseMovement(GLfloat xoffset, GLfloat yoffset, GLboolean constrainTheta) {
	xoffset *= this->MouseSensitivity;
	yoffset *= this->MouseSensitivity;

	this->Theta += yoffset*(PI / 180);
	this->Phi += xoffset*(PI / 180);

	// Make sure that when phi is out of bounds, screen doesn't get flipped
	if (constrainTheta)
	{
		if (this->Theta > 3.14f)
			this->Theta = 3.14f;
		if (this->Theta < 0.01f)
			this->Theta = 0.01f;
	}
	if (this->Phi >= 2 * PI)
		this->Phi -= 2 * PI;
	if (this->Phi <= -2 * PI)
		this->Phi += 2 * PI;

	// camUpdate camDir, camRight and camUp Vectors using the updated Eular angles
	this->updateCameraVectors();
}

void Camera::ProcessMouseScroll(GLfloat yoffset) {
	if (this->Zoom >= 1.0f && this->Zoom <= 45.0f)
		this->Zoom -= yoffset;
	if (this->Zoom <= 1.0f)
		this->Zoom = 1.0f;
	if (this->Zoom >= 45.0f)
		this->Zoom = 45.0f;
}

void Camera::updateCameraVectors() {
	// Calculate the new camDir vector
	// Modified spherical coordinate projection.
	this->camDir.x = sin(this->Theta) * cos(this->Phi);
	this->camDir.y = sin(this->Theta) * sin(this->Phi);
	this->camDir.z = cos(this->Theta);

	this->camRight.x = sin(this->Phi);
	this->camRight.y = -cos(this->Phi);
	this->camRight.z = 0.0f;

	this->camUp.x = -cos(this->Theta) * cos(this->Phi);
	this->camUp.y = -cos(this->Theta) * sin(this->Phi);
	this->camUp.z = sin(this->Theta);
}

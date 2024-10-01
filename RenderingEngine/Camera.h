#ifndef CAMERA_H
#define CAMERA_H
#pragma once

// Std. Includes
#include <vector>

// GL Includes
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace std;
using namespace glm;



// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

// Default camera values
const GLfloat PI = 3.14159265358979;
const GLfloat THETA = PI / 2.0f;
const GLfloat PHI = PI / 2.0f;
const GLfloat SPEED = 3.0f;
const GLfloat SENSITIVTY = 0.25f;
const GLfloat ZOOM = 45.0f;


// An abstract camera class that processes input and calculates the corresponding Eular Angles, Vectors and Matrices for use in OpenGL
class Camera {
public:
	// Camera Attributes
	glm::vec3 camPos;
	glm::vec3 camDir;
	glm::vec3 camUp;
	glm::vec3 camRight;
	glm::vec3 WorldcamUp;

	glm::vec3 camPosSPH;
	glm::vec3 camDirSPH;
	glm::vec3 camUpSPH;
	glm::vec3 camRightSPH;
	glm::vec3 WorldcamUpSPH;


	//Direction Angles
	GLfloat Theta;
	GLfloat Phi;
	// Camera options
	GLfloat MovementSpeed;
	GLfloat MouseSensitivity;
	GLfloat Zoom;

	GLuint WIDTH, HEIGHT;

	// Constructor with vectors
	Camera(glm::vec3 position = glm::vec3(10.0f, 10.0f, 10.0f),
		glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f),
		GLfloat theta = THETA, GLfloat phi = PHI) : camDir(glm::vec3(0.0f, 1.0f, 0.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVTY), Zoom(ZOOM)
	{
		this->camPos = position;
		this->WorldcamUp = up;
		this->Theta = theta;
		this->Phi = phi;
		this->updateCameraVectors();
	}
	// Constructor with scalar values
	Camera(GLfloat posX, GLfloat posY, GLfloat posZ, GLfloat upX, GLfloat upY, GLfloat upZ, GLfloat theta, GLfloat phi) : camDir(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVTY), Zoom(ZOOM)
	{
		this->camPos = glm::vec3(posX, posY, posZ);
		this->WorldcamUp = glm::vec3(upX, upY, upZ);
		this->Theta = theta;
		this->Phi = phi;
		this->updateCameraVectors();
	}

	// Returns the view matrix calculated using Eular Angles and the LookAt Matrix
	mat4 GetViewMatrix();

	// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
	void ProcessKeyboard(Camera_Movement direction, GLfloat deltaTime);

	// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
	void ProcessMouseMovement(GLfloat xoffset, GLfloat yoffset, GLboolean constrainTheta = true);

	// Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
	void ProcessMouseScroll(GLfloat yoffset);

private:
	// Calculates the front vector from the Camera's (updated) Eular Angles
	void updateCameraVectors();
};


#endif // !CAMERA_H
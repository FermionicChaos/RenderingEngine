#include "Test_1.h"

#include "GameState.h"

#include "Shader.h"
#include "Texture.h"
#include "Camera.h"
#include "ControlMap.h"

#include "SquareV1.h"
#include "CoordinateAxis.h"

#include "Text.h"

#include <iomanip>
#include <locale>
#include <sstream>
#include <string> 

glm::vec3 cubePositions[] = {
	glm::vec3(4.0f,  0.0f,  0.0f),
	glm::vec3(8.0f,  0.0f, 0.0f),
	glm::vec3(12.0f, 0.0f, 0.0f),
	glm::vec3(0.0f, 4.0f, 0.0f),
	glm::vec3(0.0f, 15.0f, 0.0f),
	glm::vec3(0.0f,  20.0f, 0.0f),
	glm::vec3(0.0f, 0.0f, 3.0f),
	glm::vec3(0.0f, 0.0f, 6.0f),
	glm::vec3(0.0f, 0.0f, 9.0f),
	glm::vec3(0.0f, 0.0f, 12.0f)
};

Test_1::Test_1() {

}

Test_1::~Test_1() {

}

void Test_1::ProcessKeyboard(bool * keys) {
	if (keys[GLFW_KEY_W])
		m_camera.ProcessKeyboard(FORWARD, dt);
	if (keys[GLFW_KEY_S])
		m_camera.ProcessKeyboard(BACKWARD, dt);
	if (keys[GLFW_KEY_A])
		m_camera.ProcessKeyboard(LEFT, dt);
	if (keys[GLFW_KEY_D])
		m_camera.ProcessKeyboard(RIGHT, dt);
}

void Test_1::ProcessMouseMovement(GLfloat mouse_dx, GLfloat mouse_dy) {
	m_camera.ProcessMouseMovement(mouse_dx, mouse_dy);
}

void Test_1::ProcessMouseScroll(GLfloat scroll_dx, GLfloat scroll_dy) {
	m_camera.ProcessMouseScroll(scroll_dy);
}

void Test_1::Initialize() {
	gotoState = ERROR;
	changeState = false;

	text.Initialize(WIDTH, HEIGHT);
	m_camera.WIDTH = WIDTH;
	m_camera.HEIGHT = HEIGHT;
}

void Test_1::Clear() {

}

void Test_1::Update() {

	text.setPosition(25.0f, 25.0f);
	text.scale = 0.5f;
	text.color = vec3(1.0f, 1.0f, 1.0f);
	text.text = "The quick brown fox";
}

void Test_1::Draw() {
	m_axis.Draw(m_camera);
	for (int i = 0; i < 10; i++) {
		m_square.posVec = cubePositions[i];
		m_square.Draw(m_camera);
	}
	//text.DrawText("cocks", 1.0f, vec3(1.0f, 1.0f, 1.0f));
	text.DrawText();
}


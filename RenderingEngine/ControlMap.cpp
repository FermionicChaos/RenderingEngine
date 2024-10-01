#include "ControlMap.h"

#include <iostream>

// GLEW
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace std;

#include "GameState.h"
#include "Test_1.h"
#include "Start.h"

ControlMap::ControlMap() {
	cursorEnabled = false;
	firstMouse = true;
	t0 = 0.0; t1 = 0.01;
	dt = t1 - t0;
	tdelay = 0.0;
}

ControlMap::~ControlMap() {

}

void ControlMap::keyAction(GameState * g_state) {
	//Global Controls.
	if ((keys[GLFW_KEY_F11])&&(tdelay > 1.0)) {
		if (g_state->isFullScreen) {
			g_state->isUpdateWindow = true;
			g_state->isFullScreen = false;
		}
		else {
			g_state->isUpdateWindow = true;
			g_state->isFullScreen = true;
		}
		tdelay = 0.0;
	}
	if ((keys[GLFW_KEY_F10]) && (tdelay > 1.0)) {
		if (g_state->isCursorEnabled) {
			g_state->isUpdateWindow = true;
			g_state->isCursorEnabled = false;
		}
		else {
			g_state->isUpdateWindow = true;
			g_state->isCursorEnabled = true;
		}
		tdelay = 0.0;
	}

	g_state->ProcessKeyboard(keys);

	if (activeMouse) {
		g_state->ProcessMouseMovement(mouse_xoffset, mouse_yoffset);
		activeMouse = false;
	}

	if (activeScroll) {
		g_state->ProcessMouseScroll(scroll_xoffset, scroll_yoffset);
		activeScroll = false;
	}
}

void ControlMap::keyCallback(GLFWwindow * window, int key, int scancode, int action, int mode) {
	//Senses key pressing and releasing.
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
	if (key >= 0 && key < 1024)
	{
		if (action == GLFW_PRESS)
			keys[key] = true;
		else if (action == GLFW_RELEASE)
			keys[key] = false;
	}
}

void ControlMap::mouseCallback(double xpos, double ypos) {
	if (firstMouse)
	{
		mouse_lastX = xpos;
		mouse_lastY = ypos;
		firstMouse = false;
	}

	mouse_xoffset = mouse_lastX - xpos;
	mouse_yoffset = ypos - mouse_lastY;  // Reversed since y-coordinates go from bottom to left

	mouse_lastX = xpos;
	mouse_lastY = ypos;

	activeMouse = true;
}

void ControlMap::scrollCallback(double xoffset, double yoffset) {
	scroll_xoffset = xoffset; scroll_yoffset = yoffset;

	activeScroll = true;
}

void ControlMap::Update() {
	t0 = t1;
	t1 = glfwGetTime();
	dt = t1 - t0;
	tdelay += dt;
}



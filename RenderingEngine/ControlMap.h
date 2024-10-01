#ifndef CONTROLMAP_H
#define CONTROLMAP_H

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

class ControlMap {
public:
	bool keys[1024]; //Key map;
	GLfloat mouse_lastX, mouse_lastY; //Previous mouse position.
	GLfloat mouse_xoffset, mouse_yoffset; //Mouse position difference each cycle.
	GLfloat scroll_xoffset, scroll_yoffset; //Mouse scroll offset.
	bool firstMouse;
	bool cursorEnabled; //Cursor Enabled?
	double tdelay;
	double dt, t1, t0;

	bool activeMouse;
	bool activeScroll;

	ControlMap();
	~ControlMap();

	void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode); //Activate Keys.
	void mouseCallback(double xpos, double ypos); //Cursor movement tracking.
	void scrollCallback(double xoffset, double yoffset); //Scroll movement tracking.

	void Update();

	void keyAction(GameState* g_state);
};


#endif // !CONTROLMAP_H

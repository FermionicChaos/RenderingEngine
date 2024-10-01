#include "Game.h"

#include <stdlib.h>

#include <iostream>

// GLEW
//#define GLEW_STATIC
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
// Other includes

/*
We need to find a way to make an easily writable control map for the CAMERA and CONTROL MAP to
communicate. Control map, and Camera should have distinct states for operation with eachother.

Initialize all objects once.
Process Input Events.
Update all objects based on input.
Update all time difference objects.
Draw.
Test for update.
*/

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void window_size_callback(GLFWwindow* window, int width, int height);
void do_movement(GameState* g_state);

GameState* m_game_state;
ControlMap m_controlMap;

Game::Game() {

}

Game::~Game() {
	//Clear all Device and Host memory.
	delete m_game_state;
	glfwTerminate();
}

void Game::Initialize(GLuint width, GLuint height) {
	//Initialize Context.
	// Init GLFW
	isFullScreen = false;
	isUpdateWindow = false;
	isCursorEnabled = false;

	WIDTH = width, HEIGHT = height;

	m_controlMap.mouse_lastX = WIDTH / 4;
	m_controlMap.mouse_lastY = HEIGHT / 4;

	glfwInit();
	monitor = glfwGetPrimaryMonitor();
	mode = glfwGetVideoMode(monitor);

	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create a GLFWwindow object that we can use for GLFW's functions
	window = glfwCreateWindow(WIDTH, HEIGHT, "Rendering Engine", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	// Set the required callback functions
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetWindowSizeCallback(window, window_size_callback);

	// GLFW Options
	if (!m_controlMap.cursorEnabled) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	else {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
	glewExperimental = GL_TRUE;
	// Initialize GLEW to setup the OpenGL Function pointers
	glewInit();

	// Define the viewport dimensions
	glViewport(0, 0, WIDTH, HEIGHT);

	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE); //Renders CCW/CW only surface.
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	dt = 0.0; t1 = 0.0; t0 = 0.0;
	
	m_game_state = new Start();
	m_game_state->WIDTH = WIDTH; m_game_state->HEIGHT = HEIGHT;
	m_game_state->isFullScreen = isFullScreen;
	m_game_state->isUpdateWindow = isUpdateWindow;
	m_game_state->isCursorEnabled = isCursorEnabled;
	m_game_state->Initialize();
}

void Game::run() {
	//Main Game Loop.
	while (!glfwWindowShouldClose(window))
	{
		t0 = t1;
		t1 = glfwGetTime();
		dt = t1 - t0;
		///*
		m_game_state->dt = dt;
		m_game_state->t1 = t1;
		m_game_state->t0 = t0;
		//*/
		//Change State.
		if (m_game_state->changeState) {
			m_game_state->Clear();
			WIDTH = m_game_state->WIDTH; HEIGHT = m_game_state->HEIGHT;
			isFullScreen = m_game_state->isFullScreen;
			isUpdateWindow = m_game_state->isUpdateWindow;
			isCursorEnabled = m_game_state->isCursorEnabled;
			if (m_game_state->gotoState == START) {
				delete m_game_state;
				m_game_state = new Start();
			}
			else if (m_game_state->gotoState == MAIN_MENU) {
				delete m_game_state;
				//m_game_state = new MainMenu();
			}
			else if (m_game_state->gotoState == TEST_1) {
				delete m_game_state;
				m_game_state = new Test_1();
			}
			else {
				printf("Error detected in state transition.\n");
			}
			m_game_state->changeState = false;
			m_game_state->WIDTH = WIDTH; m_game_state->HEIGHT = HEIGHT;
			m_game_state->isFullScreen = isFullScreen;
			m_game_state->isUpdateWindow = isUpdateWindow;
			m_game_state->isCursorEnabled = isCursorEnabled;
			m_game_state->Initialize();
		}
		//Resize Window.
		if (m_game_state->isUpdateWindow) {
			if (m_game_state->isFullScreen != isFullScreen) {
				if (m_game_state->isFullScreen) {
					m_controlMap.mouse_lastX = mode->width / 4;
					m_controlMap.mouse_lastY = mode->height / 4;

					glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);

					glViewport(0, 0, mode->width, mode->height);

					m_game_state->Clear();
					m_game_state->WIDTH = mode->width; m_game_state->HEIGHT = mode->height;
					m_game_state->Initialize();
				}
				else {
					m_controlMap.mouse_lastX = WIDTH / 4;
					m_controlMap.mouse_lastY = HEIGHT / 4;

					glfwSetWindowMonitor(window, NULL, WIDTH / 4, HEIGHT / 4, WIDTH, HEIGHT, mode->refreshRate);

					glViewport(0, 0, WIDTH, HEIGHT);

					m_game_state->Clear();
					m_game_state->WIDTH = WIDTH; m_game_state->HEIGHT = HEIGHT;
					m_game_state->Initialize();
				}
				cout << "FullScreen Enabled: " << m_game_state->isFullScreen << endl;
				isFullScreen = m_game_state->isFullScreen;
			}
			if (m_game_state->isCursorEnabled != isCursorEnabled) {
				if (m_game_state->isCursorEnabled) {
					glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
				}
				else {
					glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				}
				cout << "Cursor Enabled: " << m_game_state->isCursorEnabled << endl;
				isCursorEnabled = m_game_state->isCursorEnabled;
			}
			m_game_state->isUpdateWindow = false;
		}
		// Clear the colorbuffer.
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//Control Point.
		glfwPollEvents();
		do_movement(m_game_state);
		//Update all.
		m_game_state->Update();
		m_controlMap.Update();
		//Draw all.
		m_game_state->Draw();

		glfwSwapBuffers(window);
	}
}

//Moves/alters the camera positions based on user input
//We need to find a way to lend access to these variables.
//Execute actions based on input.
void do_movement(GameState* g_state) {
	m_controlMap.keyAction(g_state);
}

// (Poll Events) Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
	m_controlMap.keyCallback(window, key, scancode, action, mode);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	m_controlMap.mouseCallback(xpos, ypos);
}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	m_controlMap.scrollCallback(xoffset, yoffset);
}

void window_size_callback(GLFWwindow* window, int width, int height) {

}
#ifndef GAME_H
#define GAME_H

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

/*
These need to be dynamic for fast coding.

We need to abstract:
Control Map
Game State
Camera

*/
class Game {
private:
	//Secondary in game functions.

public:
	//Window Parameters.
	GLFWwindow* window;
	GLFWmonitor* monitor;
	const GLFWvidmode* mode;
	double dt, t1, t0;
	GLuint WIDTH, HEIGHT;
	bool isFullScreen;
	bool isUpdateWindow;
	bool isCursorEnabled;

	//Functions an initializers.
	Game();
	~Game();

	void Initialize(GLuint width,GLuint height);
	void run();

	//void Update();

};
#endif // !GAME_H


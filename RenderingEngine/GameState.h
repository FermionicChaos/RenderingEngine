#ifndef GAMESTATE_H
#define GAMESTATE_H

// GLEW
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

enum gameState {
	START,
	PROLOGUE,
	MAIN_MENU,
	TEST_1,
	
	ERROR
};

class GameState {
public:
	gameState gotoState;
	bool changeState;
	double dt, t1, t0;
	GLuint WIDTH, HEIGHT;
	bool isFullScreen;
	bool isUpdateWindow;
	bool isCursorEnabled;

	GameState();
	~GameState();

	virtual void ProcessKeyboard(bool* keys) = 0;
	virtual void ProcessMouseMovement(GLfloat mouse_dx, GLfloat mouse_dy) = 0;
	virtual void ProcessMouseScroll(GLfloat scroll_dx, GLfloat scroll_dy) = 0;

	virtual void Initialize() = 0; //Initialize all resources.
	virtual void Clear() = 0; //Clear all resources.
	virtual void Update() = 0; //
	virtual void Draw() = 0;

	/*
	gameState getState();
	bool getChange();
	GLuint getWidth();
	GLuint getHeight();
	
	void setState(gameState state);
	void setChange(bool tf);
	void setWidth(GLuint width);
	void setHeight(GLuint height);
	//*/
};

#endif // !GAMESTATE_H

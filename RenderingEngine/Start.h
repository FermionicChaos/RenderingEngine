#ifndef START_H
#define START_H

#include "GameState.h"

// GLEW
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Start : public GameState {
public:
	//Required:
	Start();
	~Start();


	// Inherited via GameState
	virtual void ProcessKeyboard(bool * keys) override;

	virtual void ProcessMouseMovement(GLfloat mouse_dx, GLfloat mouse_dy) override;

	virtual void ProcessMouseScroll(GLfloat scroll_dx, GLfloat scroll_dy) override;

	virtual void Initialize() override;

	virtual void Clear() override;

	virtual void Update() override;

	virtual void Draw() override;

	//Secondary:
};
#endif // !START_H
#include "Start.h"
#include "GameState.h"

// GLEW
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

Start::Start() {

}

Start::~Start() {

}

void Start::ProcessKeyboard(bool * keys) {

}

void Start::ProcessMouseMovement(GLfloat mouse_dx, GLfloat mouse_dy) {

}

void Start::ProcessMouseScroll(GLfloat scroll_dx, GLfloat scroll_dy) {

}

void Start::Initialize() {
	gotoState = TEST_1;
	changeState = true;
}

void Start::Clear() {

}

void Start::Update() {

}

void Start::Draw() {

}

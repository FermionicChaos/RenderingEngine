#ifndef TEST_1_H
#define TEST_1_H

#include "GameState.h"

#include "Shader.h"
#include "Texture.h"
#include "Camera.h"
#include "ControlMap.h"

#include "SquareV1.h"
#include "CoordinateAxis.h"

#include "Text.h"

class Test_1 : public GameState {
public:
	//Required:
	Test_1();
	~Test_1();

	//Secondary:
	Camera m_camera;
	Square m_square;
	CoordinateAxis m_axis;
	Text text;

	// Inherited via GameState
	virtual void ProcessKeyboard(bool * keys) override;
	virtual void ProcessMouseMovement(GLfloat mouse_dx, GLfloat mouse_dy) override;
	virtual void ProcessMouseScroll(GLfloat scroll_dx, GLfloat scroll_dy) override;
	virtual void Initialize() override;
	virtual void Clear() override;
	virtual void Update() override;
	virtual void Draw() override;
};

#endif // !TEST_1_H
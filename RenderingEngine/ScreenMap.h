#ifndef SCREENMAP_H
#define SCREENMAP_H

// GLEW
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Other includes
#include "Shader.h"
#include "Texture.h"
#include "Camera.h"
#include "cTexture.h"
using namespace std;
using namespace glm;

class ScreenMap {
public:
	Shader shader;
	Texture texture;
	GLuint VBO, VAO, EBO; //Order for somefucking reason matters, do not type "GLuint VAO, VBO" It will not work.

	cTexture GRID;
	GLuint WIDTH, HEIGHT;

	vec3 posVec;
	mat4 view, position, projection; //For some cocksucking reason, I have to declare mat4 everytime in this trash.
	GLint d_view, d_position, d_projection;

	GLfloat angle;

	ScreenMap();
	~ScreenMap();

	ScreenMap(GLuint width, GLuint height);

	void Update(double dt, Camera& camera);
	void Draw(Camera& camera);
};


#endif // !SCREENMAP_H
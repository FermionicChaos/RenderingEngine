#ifndef TEXT_H
#define TEXT_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
// GLEW
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Shader.h"

#include <ft2build.h>
#include FT_FREETYPE_H

struct Character {
	GLuint     TextureID;  // ID handle of the glyph texture
	glm::ivec2 Size;       // Size of glyph
	glm::ivec2 Bearing;    // Offset from baseline to left/top of glyph
	GLuint     Advance;    // Offset to advance to next glyph
};

class Text {
public:

	std::map<GLchar, Character> Characters;

	GLuint VAO, VBO;



	Shader shader;
	glm::mat4 projection;

	///*
	glm::vec2 pos;
	GLfloat scale;
	glm::vec3 color;
	std::string text;
	//*/
	Text();
	~Text();

	void Initialize(GLuint WIDTH, GLuint HEIGHT);
	void setPosition(GLfloat x, GLfloat y);
	void DrawText();


};

#endif // !TEXT_H


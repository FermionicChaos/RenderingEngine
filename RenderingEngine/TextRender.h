#ifndef TEXTRENDER_H
#define TEXTRENDER_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
// GLEW
//#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "stb_image.h"

#include "Shader.h"

#include <ft2build.h>
#include FT_FREETYPE_H

struct Character {
	GLuint     TextureID;  // ID handle of the glyph texture
	glm::ivec2 Size;       // Size of glyph
	glm::ivec2 Bearing;    // Offset from baseline to left/top of glyph
	GLuint     Advance;    // Offset to advance to next glyph
};

class TextRender {
public:
	FT_Library ft;
	FT_Face face;

	std::map<GLchar, Character> Characters;

	GLuint textureID;
	GLuint VAO, VBO;

	Shader shader;
	glm::mat4 projection;

	glm::vec2 pos;
	std::string text;
	GLfloat scale;
	glm::vec3 color;

	TextRender();
	~TextRender();

	void Initialize(GLuint WIDTH, GLuint HEIGHT);
	//void setPosition(GLfloat x, GLfloat y);
	//void setText(std::string input);
	//void setColor(glm::vec3 Color);
	//void DrawText();
	void DrawText(std::string text, GLfloat x, GLfloat y, GLfloat scale, glm::vec3 color);
};

#endif // !TEXTRENDER_H
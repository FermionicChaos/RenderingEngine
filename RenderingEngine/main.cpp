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
#include "stb_image.h"
using namespace std;

// Other includes
#include "Shader.h"
#include "Texture.h"
#include "Camera.h"

#include "SquareV1.h"
#include "CoordinateAxis.h"

#include "cR2.h"
#include "cTexture.h"
#include "cu_sfR2.h"

#include "Game.h"

/*
Disclaimer: I am not responsible for whatever immature comments will be found in this
rabbit hole of a project.

The final coordinate system which will use the data to compose an image.

First, in this world coordinate space, there exists objects which have a defined position in this space.
From that position vector we can relate the vertex vectors of the object to the world space
r
posVec, camPos.
First project vertex vectors in global coordinate system, then take the new vector and translate it
to camera coordinate system. After that, the camera's direction may be rotated, so using the dot product
we can project these vertex vectors into the camera's oreintation coordinate system.

The question remains, how should I group these objects? Perhaps I should group shaders, textures, VAOs and VBOs
with each individual object such as the square, and call the final draw function with the camera.

New model, Game should be a point where you can 
*/
//GLuint WIDTH = 640, HEIGHT = 480;
GLuint WIDTH = 1280, HEIGHT = 720;
//GLuint WIDTH = 1920, HEIGHT = 1080;
//GLuint WIDTH = 512, HEIGHT = 512;
//GLuint WIDTH = 1024, HEIGHT = 1024;
int main(int argc, char **argv[]) {
	if(true) {
		Game game;
		game.Initialize(WIDTH, HEIGHT);
		game.run();
	}

	return 0;
}
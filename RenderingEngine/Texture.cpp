#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <GL/glew.h>
#include "stb_image.h"

#include "Texture.h"

Texture::Texture(){
	glDeleteBuffers(1, &bufferID);
}

Texture::Texture(char * Name) {
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID); // All upcoming GL_TEXTURE_2D operations now have effect on this texture object
										   // Set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// Set texture wrapping to GL_REPEAT (usually basic wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// Set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// Load image, create texture and generate mipmaps
	unsigned char* rgb = stbi_load(Name, &width, &height, &bpp, 3);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);
	glGenerateMipmap(GL_TEXTURE_2D);
	stbi_image_free(rgb);
	glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done, so we won't accidentily mess up our texture.
}

void Texture::load(char * Name) {


	file = new char[256];
	strcpy(file, Name);
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID); // All upcoming GL_TEXTURE_2D operations now have effect on this texture object
											 // Set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// Set texture wrapping to GL_REPEAT (usually basic wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// Set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// Load image, create texture and generate mipmaps
	unsigned char* rgba = stbi_load(Name, &width, &height, &bpp, 4);

	glGenBuffers(GL_PIXEL_UNPACK_BUFFER_ARB, &bufferID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*4, rgba, GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgba);
	glGenerateMipmap(GL_TEXTURE_2D);
	stbi_image_free(rgba);
	glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done, so we won't accidentily mess up our texture.
}

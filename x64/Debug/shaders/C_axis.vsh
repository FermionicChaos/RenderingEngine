#version 330 core
layout (location = 0) in vec3 position;
layout (location = 2) in vec3 rgb;

out vec3 ourColor;

uniform mat4 posVec;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * posVec * vec4(position, 1.0f);
	ourColor = rgb;
}
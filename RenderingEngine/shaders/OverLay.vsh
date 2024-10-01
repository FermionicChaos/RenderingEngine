#version 330 core
layout (location = 0) in vec3 position;

out vec2 TexCoord;

void main()
{
    //gl_Position = d_projection * d_view * d_position * vec4(local_vertex, 1.0f);
	gl_Position = vec4(position.x, position.y, position.z, 1.0);
	TexCoord = vec2(
	(position.x + 1)/2,
	1 - (position.y + 1)/2
	);
}
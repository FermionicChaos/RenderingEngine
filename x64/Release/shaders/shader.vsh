#version 330 core
layout (location = 0) in vec3 local_vertex;
layout (location = 2) in vec2 texCoord;

out vec2 TexCoord;

uniform mat4 d_position;
uniform mat4 d_view;
uniform mat4 d_projection;

void main()
{
    gl_Position = d_projection * d_view * d_position * vec4(local_vertex, 1.0f);
    TexCoord = vec2(texCoord.x, 1.0 - texCoord.y);
}
#version 330 core
in vec2 TexCoords;

out vec4 color;

uniform vec2 resolution;
uniform sampler2D TextureUnit0;

void main() {    
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(TextureUnit0, TexCoords).r);
    color = vec4(textColor, 1.0) * sampled;
}  
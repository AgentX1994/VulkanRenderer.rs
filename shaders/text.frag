#version 450
layout (location=0) in vec2 in_tex_coord;
layout (location=1) in vec3 in_color;

layout (location=0) out vec4 color;

layout(set=0,binding=0) uniform sampler2D font_atlas;

void main() {
    color = vec4(in_color, texture(font_atlas, in_tex_coord));
}
#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout (location=0) in vec2 in_tex_coord;
layout (location=1) in vec3 in_color;
layout (location=2) flat in uint in_texture_id;

layout (location=0) out vec4 color;

layout(set=0,binding=0) uniform sampler2D letter_textures[];

void main() {
    color = vec4(in_color, texture(letter_textures[in_texture_id], in_tex_coord));
}
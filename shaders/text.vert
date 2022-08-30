#version 450
layout (location=0) in vec3 in_position;
layout (location=1) in vec2 in_tex_coord;
layout (location=2) in vec3 in_color;
layout (location=3) in uint in_texture_id;

layout (location=0) out vec2 out_tex_coord;
layout (location=1) out vec3 out_color;
layout (location=2) out uint out_texture_id;

void main() {
    gl_Position = vec4(in_position, 1.0);
    out_tex_coord = in_tex_coord;
    out_color = in_color;
    out_texture_id = in_texture_id;
}
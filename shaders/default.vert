#version 450

layout (location=0) in vec3 position;
layout (location=1) in vec3 color;
layout (location=2) in vec2 uv;
layout (location=3) in mat4 model_matrix;
layout (location=7) in vec3 color_mod;

layout (location=0) out vec4 outColor;

void main() {
    // gl_Position = positions[gl_VertexIndex];
    gl_Position = model_matrix*vec4(position, 1.0);
    // gl_PointSize = size;
    outColor = vec4(color + color_mod, 1.0);
}
#version 450

layout (location=0) in vec3 position;
layout (location=1) in vec3 color;
layout (location=2) in vec2 uv;
layout (location=3) in mat4 model_matrix;
layout (location=7) in vec3 color_mod;

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 view_matrix;
} ubo;

layout (location=0) out vec4 outColor;

void main() {
    gl_Position = ubo.view_matrix*model_matrix*vec4(position, 1.0);
    outColor = vec4(color + color_mod, 1.0);
}
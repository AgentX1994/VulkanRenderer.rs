#version 450

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec3 color;
layout (location=3) in vec2 uv;
layout (location=4) in mat4 model_matrix;
layout (location=8) in vec3 color_mod;

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 view_matrix;
    mat4 projection_matrix;
} ubo;

layout (location=0) out vec3 outColor;
layout (location=1) out vec3 out_normal;

void main() {
    gl_Position = ubo.projection_matrix*ubo.view_matrix*model_matrix*vec4(position, 1.0);
    outColor = color + color_mod;
    out_normal = vec3(transpose(inverse(model_matrix))*vec4(normalize(normal), 0.0));
}
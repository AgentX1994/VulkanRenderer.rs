#version 450

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 uv;
layout (location=3) in mat4 model_matrix;
layout (location=7) in mat4 inverse_model_matrix;
layout (location=11) in vec3 color;
layout (location=12) in float metallic_in;
layout (location=13) in float roughness_in;
layout (location=14) in uint texture_id_in;

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 view_matrix;
    mat4 projection_matrix;
} ubo;

layout (location=0) out vec3 outColor;
layout (location=1) out vec3 out_normal;
layout (location=2) out vec4 worldpos;
layout (location=3) out vec3 camera_pos;
layout (location=4) out float metallic;
layout (location=5) out float roughness;
layout (location=6) out vec2 uv_out;
layout (location=7) flat out uint texture_id;

void main() {
    worldpos = model_matrix*vec4(position, 1.0);
    gl_Position = ubo.projection_matrix*ubo.view_matrix*worldpos;
    camera_pos =
	- ubo.view_matrix[3][0] * vec3 (ubo.view_matrix[0][0],ubo.view_matrix[1][0],ubo.view_matrix[2][0])
	- ubo.view_matrix[3][1] * vec3 (ubo.view_matrix[0][1],ubo.view_matrix[1][1],ubo.view_matrix[2][1])
	- ubo.view_matrix[3][2] * vec3 (ubo.view_matrix[0][2],ubo.view_matrix[1][2],ubo.view_matrix[2][2]);

    outColor = color;
    out_normal = vec3(transpose(inverse_model_matrix)*vec4(normalize(normal), 0.0));
    metallic = metallic_in;
    roughness = roughness_in;
    uv_out = uv;
    texture_id = texture_id_in;
}
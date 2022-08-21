#version 450

layout (location=0) in vec3 color;
layout (location=1) in vec3 normal_varied;
layout (location=0) out vec4 outColor;

void main() {
    vec3 direction_to_light = normalize(vec3(-1, -1, 0));
    vec3 normal = normalize(normal_varied);
    vec3 ambient = color;
    vec3 diffuse = max(dot(normal, direction_to_light), 0) * color;
    outColor = vec4(0.5*ambient + 0.5*diffuse, 1);
}
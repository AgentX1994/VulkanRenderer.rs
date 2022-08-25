#version 450

layout (location=0) in vec3 color;
layout (location=1) in vec3 normal_varied;
layout (location=2) in vec4 worldpos;
layout (location=0) out vec4 outColor;

const float PI = 3.14159265358979323846264;

struct DirectionalLight {
    vec3 direction_to_light;
    vec3 irradiance;
};

struct PointLight {
    vec3 position;
    vec3 luminous_flux;
};

const int NUMBER_OF_POINTLIGHTS = 3;
PointLight point_lights[NUMBER_OF_POINTLIGHTS] = {
    PointLight(vec3(1.5, 0.0, 0.0), vec3(10, 10, 10)),
    PointLight(vec3(1.5, 0.2, 0.0), vec3(5, 5, 5)),
    PointLight(vec3(1.6, -0.2, 0.1), vec3(5, 5, 5)),
};

vec3 compute_radiance(vec3 irradiance, vec3 light_dir, vec3 normal, vec3 surface_color) {
    return irradiance*(max(dot(normal, light_dir), 0))*surface_color;
}

vec3 tone_map(vec3 total_radiance) {
    return total_radiance / (1 + total_radiance);
}

void main() {
    vec3 total_radiance = vec3(0);
    vec3 normal = normalize(normal_varied);

    DirectionalLight d_light = DirectionalLight(normalize(vec3(-1, -1, 0)), vec3(0.1, 0.1, 0.1));
    total_radiance += compute_radiance(d_light.irradiance, d_light.direction_to_light, normal, color);

    for (int i = 0; i < NUMBER_OF_POINTLIGHTS; ++i) {
        PointLight light = point_lights[i];
        vec3 direction_to_light = normalize(light.position - worldpos.xyz);
        float d = length(worldpos.xyz - light.position);
        vec3 irradiance = light.luminous_flux/(4*PI*d*d);

        total_radiance += compute_radiance(irradiance, direction_to_light, normal, color);
    }

    outColor = vec4(tone_map(total_radiance), 1);
}
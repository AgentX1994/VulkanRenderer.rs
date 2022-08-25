#version 450

layout (location=0) in vec3 color;
layout (location=1) in vec3 normal_varied;
layout (location=2) in vec4 worldpos;
layout (location=3) in vec3 camera_pos;
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

float distribution(vec3 normal, vec3 halfvector, float roughness) {
    float NdotH = dot(halfvector, normal);
    if (NdotH > 0) {
        float r = roughness * roughness;
        return r / (PI * (1 + NdotH*NdotH*(r-1))*(1 + NdotH*NdotH*(r-1)));
    } else {
        return 0.0;
    }
}

float geometry(vec3 light, vec3 normal, vec3 view, float roughness) {
    float NdotL = abs(dot(normal, light));
    float NdotV = abs(dot(normal, view));
    return 0.5 / max(0.01, mix(2*NdotL*NdotV, NdotL+NdotV, roughness));
}

vec3 compute_radiance(vec3 irradiance, vec3 light_dir, vec3 normal, vec3 camera_dir, vec3 surface_color) {
    float NdotL = max(dot(normal, light_dir), 0);

    vec3 irradiance_on_surface = irradiance*NdotL;

    float metallic = 1.0;
    float roughness = 0.5;
    roughness *= roughness;

    vec3 F0 = mix(vec3(0.03), surface_color, vec3(metallic));
    vec3 reflected_irradiance = (F0 + (1 - F0)*(1-NdotL)*(1-NdotL)*(1-NdotL)*(1-NdotL)*(1-NdotL)) * irradiance_on_surface;
    vec3 refracted_irradiance = irradiance_on_surface - reflected_irradiance;
    vec3 refracted_not_absorbed_irradiance = refracted_irradiance * (1-metallic);

    vec3 halfvector = normalize(0.5*(camera_dir + light_dir));
    float NdotH = max(dot(normal, halfvector), 0);
    vec3 F = (F0 + (1 - F0)*(1 - NdotH)*(1 - NdotH)*(1 - NdotH)*(1 - NdotH)*(1 - NdotH));
    vec3 relevant_reflection = reflected_irradiance*F*geometry(light_dir, normal, camera_dir, roughness) * distribution(normal, halfvector, roughness);
    return refracted_not_absorbed_irradiance*surface_color/PI + relevant_reflection;
}

vec3 tone_map(vec3 total_radiance) {
    return total_radiance / (1 + total_radiance);
}

void main() {
    vec3 total_radiance = vec3(0);
    vec3 normal = normalize(normal_varied);
    vec3 direction_to_camera = normalize(camera_pos - worldpos.xyz);

    DirectionalLight d_light = DirectionalLight(normalize(vec3(-1, -1, 0)), vec3(0.1, 0.1, 0.1));
    total_radiance += compute_radiance(d_light.irradiance, d_light.direction_to_light, normal, direction_to_camera, color);

    for (int i = 0; i < NUMBER_OF_POINTLIGHTS; ++i) {
        PointLight light = point_lights[i];
        vec3 direction_to_light = normalize(light.position - worldpos.xyz);
        float d = length(worldpos.xyz - light.position);
        vec3 irradiance = light.luminous_flux/(4*PI*d*d);

        total_radiance += compute_radiance(irradiance, direction_to_light, normal, direction_to_camera, color);
    }

    outColor = vec4(tone_map(total_radiance), 1);
}
#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout (location=0) in vec3 normal_varied;
layout (location=1) in vec4 worldpos;
layout (location=2) in vec3 camera_pos;
layout (location=3) in vec2 uv;

layout (location=0) out vec4 outColor;

readonly layout (set=1, binding=0) buffer StorageBufferObject {
    float num_directional;
    float num_point;
    vec3 data[];
} sbo;

layout (set=2, binding=0) uniform sampler2D texture_sampler;

layout (set=2, binding=1) uniform MaterialParameters {
    float metallic;
    float roughness;
} material_parameters;

const float PI = 3.14159265358979323846264;

struct DirectionalLight {
    vec3 direction_to_light;
    vec3 irradiance;
};

struct PointLight {
    vec3 position;
    vec3 luminous_flux;
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

vec3 compute_radiance(vec3 irradiance, vec3 light_dir, vec3 normal, vec3 camera_dir, vec3 surface_color, float metallic, float roughness) {
    float NdotL = max(dot(normal, light_dir), 0);

    vec3 irradiance_on_surface = irradiance*NdotL;

    roughness = roughness * roughness;

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
    int num_dir = int(sbo.num_directional);
    int num_point = int(sbo.num_point);

    vec3 surface_color = texture(texture_sampler, uv).rgb;

    for (int i = 0; i < num_dir; i++) {
        vec3 data1 = sbo.data[2*i];
        vec3 data2 = sbo.data[2*i+1];
        DirectionalLight d_light = DirectionalLight(normalize(data1), data2);

        total_radiance += compute_radiance(
            d_light.irradiance,
            d_light.direction_to_light,
            normal,
            direction_to_camera,
            surface_color,
            material_parameters.metallic,
            material_parameters.roughness);
    }

    for (int i = 0; i < num_point; i++) {
        vec3 data1 = sbo.data[2*i + 2*num_dir];
        vec3 data2 = sbo.data[2*i + 1 + 2*num_dir];
        PointLight light = PointLight(data1, data2);

        vec3 direction_to_light = normalize(light.position - worldpos.xyz);
        float d = length(worldpos.xyz - light.position);
        vec3 irradiance = light.luminous_flux/(4*PI*d*d);

        total_radiance += compute_radiance(
            irradiance,
            direction_to_light,
            normal,
            direction_to_camera,
            surface_color,
            material_parameters.metallic,
            material_parameters.roughness);
    }

    outColor = vec4(tone_map(total_radiance), 1);
}

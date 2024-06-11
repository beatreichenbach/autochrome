// https://google.github.io/filament/Filament.html

float D_GGX(float NoH, float roughness) {
	// Trowbridge-Reitz Normal Distribution Function (GGX)
	// NoH: dot(n, h) where n: normal and h: 
    float a2 = roughness * roughness;
    float f = NoH * NoH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * f * f);
}

float3 F_Schlick(float u, float3 f0) {
	// Fresnel Function
	// f0: specular reflectance at normal incidence
	// u: incident angle
    return f0 + ((float3) (1.0f) - f0) * pow(1.0f - u, 5.0f);
}

float V_SmithGGXCorrelated(float NoV, float NoL, float roughness) {
	// Visibility Function
    float a2 = roughness * roughness;
    float GGXL = NoV * sqrt((-NoL * a2 + NoL) * NoL + a2);
    float GGXV = NoL * sqrt((-NoV * a2 + NoV) * NoV + a2);
    return 0.5f / (GGXV + GGXL);
}

float Fd_Lambert() {
	// Simple Lambertian BRDF
    return 1.0f / M_PI_F;
}

__kernel void BRDF(
    write_only image2d_t output,
    float perceptualRoughness,
    float height
) {
    int2 p;
    p.x = get_global_id(0);
    p.y = get_global_id(1);

    int2 size;
    size.x = get_global_size(0);
    size.y = get_global_size(1);

    const float3 specular_albedo = (float3) (1, 1, 1);
    const float3 normal = (float3) (0, 1, 0);

	float3 position;
	position.x = (float) p.x / (size.x - 1);
	position.y = 0;
	position.z = (float) p.y / (size.y - 1);

	float3 camera_position = (float3) (0.5f, height, 0.5f);

	float3 light_position3 = (float3) (0.5f, 1.0f, 0.5f);

	float3 light_dir = normalize(light_position3 - position);
	float3 view_dir = normalize(camera_position - position);

	// view_dir = (float3) (0, 1, 0);

    float3 half_vec = normalize(view_dir + light_dir);
    float NoV = fabs(dot(normal, view_dir)) + 1e-5;
    float NoL = clamp(dot(normal, light_dir), 0.0f, 1.0f);
    float NoH = clamp(dot(normal, half_vec), 0.0f, 1.0f);
    float LoH = clamp(dot(light_dir, half_vec), 0.0f, 1.0f);

    // perceptually linear roughness to roughness
    float roughness = perceptualRoughness * perceptualRoughness;

    float D = D_GGX(NoH, roughness);
    float3  F = F_Schlick(LoH, specular_albedo);
    float V = V_SmithGGXCorrelated(NoV, NoL, roughness);

    // specular BRDF
    float3 Fr = (D * V) * F;
    float4 rgba = (float4) (Fr, 1);

    write_imagef(output, p, rgba);
}

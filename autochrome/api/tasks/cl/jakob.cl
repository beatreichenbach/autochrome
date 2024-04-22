int find_interval(
    __constant float* values,
    const uint resolution,
    const float x
)
{
    // returns the index of value in values closest to x
    int left = 0;
    int last_interval = resolution - 2;
    int size = last_interval;

    while (size > 0) {
        int half_ = size >> 1;
        int middle = left + half_ + 1;
        if (values[middle] <= x) {
            left = middle;
            size -= half_ + 1;
        } else {
            size = half_;
        }
    }
    int interval = min((int) last_interval, left);
    return interval;
}

float4 fetch(
    const float4 xyz,
    __constant float* model,
    __constant float* scale,
    const uint resolution
) {
    // find the the coefficients by interpolating the value from the model of xyz

    // find largest value in xyz
    uchar i = 0;
    for (uchar j = 1; j < 3; ++j) {
        if (xyz[j] >= xyz[i]) {
            i = j;
        }
    }

    float z = xyz[i];
    // Prevent nan values for (0, 0, 0)
    // not sure what normalize is for, originally called scale
    float normalize;
    if (z > 0) {
        normalize = (resolution - 1) / z;
    } else {
        normalize = 0;
    }
    // float normalize = z > 0 ? (resolution - 1) / z : 0;
    float x = xyz[(i + 1) % 3] * normalize;
    float y = xyz[(i + 2) % 3] * normalize;

    // Trilinearly interpolated lookup
    uint xi = min((uint) x, resolution - 2);
    uint yi = min((uint) y, resolution - 2);
    uint zi = (uint) find_interval(scale, resolution, z);

    uint offset = (((i * resolution + zi) * resolution + yi) * resolution + xi) * COEFFICIENTS_COUNT;

    uint dx = COEFFICIENTS_COUNT;
    uint dy = COEFFICIENTS_COUNT * resolution;
    uint dz = COEFFICIENTS_COUNT * resolution * resolution;

    float x1 = x - xi;
    float x0 = 1 - x1;

    float y1 = y - yi;
    float y0 = 1 - y1;

    float z1 = (z - scale[zi]) / (scale[zi + 1] - scale[zi]);
    float z0 = 1 - z1;

    float4 coefficients = 0;
    for (uchar j = 0; j < COEFFICIENTS_COUNT; ++j) {
        float tmp1 = (model[offset] * x0 + model[offset + dx] * x1) * y0;
        tmp1 += (model[offset + dy] * x0 + model[offset + dy + dx] * x1) * y1;
        tmp1 *= z0;

        float tmp2 = (model[offset + dz] * x0 + model[offset + dz + dx] * x1) * y0;
        tmp2 += (model[offset + dz + dy] * x0 + model[offset + dz + dy + dx] * x1) * y1;
        tmp2 *= z1;
        // TODO: this somehow works??? since when can we assign float4 vars this way? dafuq?
        coefficients[j] = tmp1 + tmp2;
        offset++;
    }
    return coefficients;
}


// Evaluate a polynomial with coefficients at a specific wavelength
float eval_precise(float4 coefficients, float wavelength) {
    // Get spectral value for wavelength based on coefficients
    float tmp = fma(coefficients[0], wavelength, coefficients[1]);
    float x = fma(tmp, wavelength, coefficients[2]);
    float y = 1.0f / sqrt(fma(x, x, 1.0f));
    return fma(0.5f * x, y, 0.5f);
}

__kernel void xyz_to_xyz(
    read_only image2d_t input,
    write_only image2d_t output,
    __constant float* lambdas,
    __constant float4* cmfs,
    __constant float* illuminant,
    __constant float* model,
    __constant float* scale,
    const uint resolution
) {
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 xyz = read_imagef(input, sampler, (int2)(x, y));
    // xyz = (float4) (0.0F, 0.01F, 0.0F, 1);

    float4 coefficients = fetch(xyz, model, scale, resolution);

    // TODO: k is a shit name
    float4 k = 0;
    float4 xyz_reconstructed = 0; 
    for (uint i = 0; i < LAMBDA_COUNT; ++i) {
        float spectrum_value = eval_precise(coefficients, lambdas[i]);
        float4 a = cmfs[i] * illuminant[i];
        k += a;
        xyz_reconstructed += a * spectrum_value;
    }

    xyz_reconstructed /= k;
    xyz_reconstructed.w = 0;
    float4 rgba = fabs(xyz - xyz_reconstructed);
    write_imagef(output, (int2)(x, y), rgba);
}

__kernel void xyz_to_sd(
    read_only image2d_t image,
    __global float* sd,
    __constant float* lambdas,
    __constant float4* cmfs,
    __constant float* illuminant,
    __constant float* model,
    __constant float* scale,
    const uint resolution
) {
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 size = get_image_dim(image);

    float4 xyz = read_imagef(image, sampler, (int2)(x, y));
    float4 coefficients = fetch(xyz, model, scale, resolution);

    // TODO: k is a shit name
    float4 k = 0;
    float4 xyz_reconstructed = 0; 
    for (uint i = 0; i < LAMBDA_COUNT; ++i) {
        float spectrum_value = eval_precise(coefficients, lambdas[i]);
        float4 a = cmfs[i] * illuminant[i];
        k += a;
        xyz_reconstructed += a * spectrum_value;
    }

    xyz_reconstructed /= k;
    xyz_reconstructed.w = 0;



    float4 rgba = fabs(xyz - xyz_reconstructed);
    // write_imagef(output, (int2)(x, y), rgba);
}

#define EPSILON 0.00001f

__kernel void xyz_to_mask(
    read_only image2d_t image,
    write_only image2d_t output_r,
    write_only image2d_t output_g,
    write_only image2d_t output_b,
    __constant float* lambdas,
    __constant float4* cmfs,
    __constant float* illuminant,
    __constant float* model,
    __constant float* scale,
    const uint model_resolution,
    __constant float4* sensitivity
) {
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 uv = (int2) (x, y);

    float4 xyz = read_imagef(image, sampler, uv);

    if (xyz.x < EPSILON || xyz.y < EPSILON || xyz.z < EPSILON) {
        write_imagef(output_r, (int2)(x, y), (float4) 0);
        write_imagef(output_g, (int2)(x, y), (float4) 0);
        write_imagef(output_b, (int2)(x, y), (float4) 0);
        return;
    }

    float4 coefficients = fetch(xyz, model, scale, model_resolution);

    float4 integral = 0;

    float4 red_layer = 0;
    float4 green_layer = 0;
    float4 blue_layer = 0;

    // float4 emulsion = 0; 
    for (uint i = 0; i < LAMBDA_COUNT; ++i) {
        float lambda_rel = (float) i / (LAMBDA_COUNT - 1);
        float spectrum_value = eval_precise(coefficients, lambda_rel);

        float sum = sensitivity[i].x + sensitivity[i].y + sensitivity[i].z;
        float4 ratio = 0;
        if (sum != 0) {
            ratio = sensitivity[i] /  sum;
        }

        integral += cmfs[i] * illuminant[i];
        // emulsion += cmfs[i] * illuminant[i] * spectrum_value;

        red_layer += cmfs[i] * illuminant[i] * spectrum_value * ratio.x;
        green_layer += cmfs[i] * illuminant[i] * spectrum_value * ratio.y;
        blue_layer += cmfs[i] * illuminant[i] * spectrum_value * ratio.z;

    }
    integral.w = 1;

    // emulsion /= integral;


    red_layer /= integral;
    blue_layer /= integral;
    green_layer /= integral;

    // float4 emulsion = red_layer + green_layer + blue_layer;
    // emulsion = fabs(xyz - emulsion);
    
    write_imagef(output_r, (int2)(x, y), red_layer);
    write_imagef(output_g, (int2)(x, y), green_layer);
    write_imagef(output_b, (int2)(x, y), blue_layer);
}

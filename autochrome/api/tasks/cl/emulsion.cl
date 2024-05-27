#define EPSILON 0.00001f

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
        int half_size = size >> 1;
        int middle = left + half_size + 1;
        if (values[middle] <= x) {
            left = middle;
            size -= half_size + 1;
        } else {
            size = half_size;
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
    // TODO: not sure what normalize is for, originally called scale
    // Prevent nan values for (0, 0, 0)
    float normalize;
    if (z > 0) {
        normalize = (resolution - 1) / z;
    } else {
        normalize = 0;
    }
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

    float xd = x - xi;
    float yd = y - yi;
    float zd = (z - scale[zi]) / (scale[zi + 1] - scale[zi]);

    // NOTE: Trilateral interpolation
    //       https://en.wikipedia.org/wiki/Trilinear_interpolation
    float4 coefficients = 0;
    for (uchar j = 0; j < COEFFICIENTS_COUNT; ++j) {
        float c000 = model[offset];
        float c100 = model[offset + dx];
        float c010 = model[offset + dy];
        float c001 = model[offset + dz];
        float c110 = model[offset + dx + dy];
        float c101 = model[offset + dx + dz];
        float c011 = model[offset + dy + dz];
        float c111 = model[offset + dx + dy + dz];

        float c00 = c000 * (1 - xd) + c100 * xd;
        float c01 = c001 * (1 - xd) + c101 * xd;
        float c10 = c010 * (1 - xd) + c110 * xd;
        float c11 = c011 * (1 - xd) + c111 * xd;

        float c0 = c00 * (1 - yd) + c10 * yd;
        float c1 = c01 * (1 - yd) + c11 * yd;

        float c = c0 * (1 - zd) + c1 * zd;
        coefficients[j] = c;
        offset++;
    }


    // float x1 = x - xi;
    // float x0 = 1 - x1;

    // float y1 = y - yi;
    // float y0 = 1 - y1;

    // float z1 = (z - scale[zi]) / (scale[zi + 1] - scale[zi]);
    // float z0 = 1 - z1;

    // float4 coefficients = 0;
    // for (uchar j = 0; j < COEFFICIENTS_COUNT; ++j) {
    //     float tmp1 = (model[offset] * x0 + model[offset + dx] * x1) * y0;
    //     tmp1 += (model[offset + dy] * x0 + model[offset + dy + dx] * x1) * y1;
    //     tmp1 *= z0;

    //     float tmp2 = (model[offset + dz] * x0 + model[offset + dz + dx] * x1) * y0;
    //     tmp2 += (model[offset + dz + dy] * x0 + model[offset + dz + dy + dx] * x1) * y1;
    //     tmp2 *= z1;
    //     // TODO: this somehow works??? since when can we assign float4 vars this way? dafuq?
    //     coefficients[j] = tmp1 + tmp2;
    //     offset++;
    // }

    return coefficients;
}


float eval_precise(float4 coefficients, float wavelength) {
    // get spectral value for lambda based on coefficients

    float c = fma(coefficients[0], wavelength, coefficients[1]);
    float x = fma(c, wavelength, coefficients[2]);
    float y = 1.0f / sqrt(fma(x, x, 1.0f));
    return fma(0.5f * x, y, 0.5f);
}



__constant sampler_t SAMPLER_ABS = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void emulsion_layers(
    read_only image2d_t xyz_image,
    write_only image2d_t spectral_r,
    write_only image2d_t spectral_g,
    write_only image2d_t spectral_b,
    __constant float* model,
    const uint model_resolution,
    __constant float* scale,
    __constant float* lambdas,
    __constant float4* cmfs,
    __constant float* illuminant,
    __constant float4* sensitivity
) {
    int2 uv;
    uv.x = get_global_id(0);
    uv.y = get_global_id(1);

    float4 xyz = read_imagef(xyz_image, SAMPLER_ABS, uv);

    if (xyz.x < EPSILON || xyz.y < EPSILON || xyz.z < EPSILON) {
        write_imagef(spectral_r, uv, (float4) 0);
        write_imagef(spectral_g, uv, (float4) 0);
        write_imagef(spectral_b, uv, (float4) 0);
        return;
    }

    float4 coefficients = fetch(xyz, model, scale, model_resolution);

    float4 integral = 0;

    float4 red_layer = 0;
    float4 green_layer = 0;
    float4 blue_layer = 0;

    for (uint i = 0; i < LAMBDA_COUNT; ++i) {
        float lambda_rel = (float) i / (LAMBDA_COUNT - 1);
        float spectrum_value = eval_precise(coefficients, lambda_rel);

        float sum = sensitivity[i].x + sensitivity[i].y + sensitivity[i].z;
        float4 ratio = 0;
        if (sum != 0) {
            ratio = sensitivity[i] /  sum;
        }

        integral += cmfs[i] * illuminant[i];

        red_layer += cmfs[i] * illuminant[i] * spectrum_value * ratio.x;
        green_layer += cmfs[i] * illuminant[i] * spectrum_value * ratio.y;
        blue_layer += cmfs[i] * illuminant[i] * spectrum_value * ratio.z;

    }
    integral.w = 1;

    red_layer /= integral;
    blue_layer /= integral;
    green_layer /= integral;
    
    write_imagef(spectral_r, uv, red_layer);
    write_imagef(spectral_g, uv, green_layer);
    write_imagef(spectral_b, uv, blue_layer);
}

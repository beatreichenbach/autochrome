// __kernel void convolve_gauss_blur_2D_image(
//     __read_only image2d_t srcImg,
//     __write_only image2d_t dstImag,
//     int width, int height,
//     __constant float *filter,
//     int half_size
// ) {
//     int2 pos = {get_global_id(0), get_global_id(1)};

//     int2 filter_size = get_image_dim(filter);

//     float sum = 0.0f;

//     int2 coord;

//     for (int x = 0; x < filter_size.x; x++)
//         for (int y = 0; y < filter_size.y; y++) {
//             coord = (int2)(pos.x + x - half_size, pos.y + y - half_size);
//             sum += filter[y * (2 * half_size + 1) + x] * read_imagef(srcImg, sampler_im, coord).x;
//         }

//     write_imagef(dstImag, pos, sum);
// }


__kernel void ConvolveH(
    __read_only image2d_t inputImage,
    __write_only image2d_t outputImage
,    __constant float* mask,
    int mask_size
) {
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    float4 rgba_in = (float4)(0,0,0,0);
    float4 rgba_out = (float4)(0,0,0,0);
    for(int mask_index = -mask_size; mask_index < mask_size+1; ++mask_index) {
        rgba_in = read_imagef(inputImage, sampler, pos + (int2)(mask_index, 0));
        rgba_out += rgba_in * mask[mask_size + mask_index];
    }
    write_imagef(outputImage, pos, rgba_out);
}

__kernel void ConvolveV(
    __read_only image2d_t inputImage,
    __write_only image2d_t outputImage,
    __constant float* mask,
    int mask_size
) {
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    float4 rgba_in = (float4)(0,0,0,0);
    float4 rgba_out = (float4)(0,0,0,0);  
    for(int mask_index = -mask_size; mask_index < mask_size+1; ++mask_index) {
        rgba_in = read_imagef(inputImage, sampler, pos + (int2)(0, mask_index));
        rgba_out += rgba_in * mask[mask_size + mask_index];
    }
    write_imagef(outputImage, pos, rgba_out);
}
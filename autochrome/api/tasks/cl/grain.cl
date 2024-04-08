#define EPSILON_GREY_LEVEL 0.1f
#define MAX_GREY_LEVEL 255

/*
* @brief Square distance
*
* @param lambda parameter of the Poisson process
* @param x1, y1 : x, y coordinates of the first point
* @param x2, y2 : x, y coordinates of the second point
* @return squared Euclidean distance
*/
// double square_distance(
//     const double x1,
//     const double y1,
//     const double x2,
//     const double y2
//     )
// {
//     return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
// }
double square_distance(const double2 a, const double2 b)
{
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}
    
__kernel void grain(
    read_only image2d_t image,
    write_only image2d_t output,
    __constant float2* lambda_lut,
    const int samples,
    const float grain_sigma,
    const float grain_radius,
    const float blur_sigma,
    const uint seed_offset,
    const float4 render_bounds
)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

    int2 p;
    p.x = get_global_id(0);
    p.y = get_global_id(1);

    int2 output_size = get_image_dim(output);

    float2 bounds_scale; 
    bounds_scale.x = (float)(output_size.x - 1) / (render_bounds.z - render_bounds.x);
    bounds_scale.y = (float)(output_size.y - 1) / (render_bounds.w - render_bounds.y);

    float cell_size = 1 / ceil(1 / grain_radius);
    float grain_radius_squared = grain_radius * grain_radius;
    float mu, sigma;
    float grain_radius_max = grain_radius;

    //calculate the mu and sigma for the lognormal distribution
    if (grain_sigma > 0.0) {
        sigma = sqrt(log((grain_sigma / grain_radius) * (grain_sigma / grain_radius) + 1.0f));
        mu = log(grain_radius) - (sigma * sigma) / 2.0f;
        float normal_quantile = 3.0902; //standard normal quantile for alpha=0.999
        float log_normal_quantile = exp(mu + sigma * normal_quantile);
        grain_radius_max = log_normal_quantile;
    }


    float value = 0;

    //conversion from output grid (xOut,yOut) to input grid (xIn,yIn)
    //we inspect the middle of the output pixel (1/2)
    //the size of a pixel is (render_bounds.z - render_bounds.x) / output_size.x

    float2 render_p;
    render_p.x = render_bounds.x + ((float) p.x + 0.5f) * ((render_bounds.z - render_bounds.x) / (float) output_size.x);
    render_p.y = render_bounds.y + ((float) p.y + 0.5f) * ((render_bounds.w - render_bounds.y) / (float) output_size.y);

    // Simulate Poisson process on the 4 neighborhood cells of (x,y)
    uint p_monte_carlo = pcg_hash((uint) 2016 * seed_offset);
    // init_rand(&p_monte_carlo, (uint) 2016 * seed_offset);

    float4 rgba;
    uint4 bounds;
    float2 gaussian;
    for (int i = 0; i < samples; i++) {
        //// this is the pos of the sample in ndc space 
        gaussian.x = render_p.x + blur_sigma * rand_gaussian(&p_monte_carlo) / bounds_scale.x;
        gaussian.y = render_p.y + blur_sigma * rand_gaussian(&p_monte_carlo) / bounds_scale.y;

        // Determine the bounding boxes around the current shifted pixel
        // these operations are set to double precision because the number of cells can be quite large
        bounds.x = (uint) floor(((double) gaussian.x - (double) grain_radius_max) / (double) cell_size);
        bounds.y = (uint) floor(((double) gaussian.y - (double) grain_radius_max) / (double) cell_size);
        bounds.z = (uint) floor(((double) gaussian.x + (double) grain_radius_max) / (double) cell_size);
        bounds.w = (uint) floor(((double) gaussian.y + (double) grain_radius_max) / (double) cell_size);

        bool point_covered = false; // used to break all for loops





        for(uint cell_x = bounds.x; cell_x <= bounds.z; cell_x++) { // x-cell number
            if(point_covered == true) {
                break;
            }

            for(uint cell_y = bounds.y; cell_y <= bounds.w; cell_y++) { // y-cell number
                if(point_covered == true) {
                    break;
                }

                double2 cell_corner = (double2) (cell_x, cell_y) * cell_size;

                uint seed = cell_seed(cell_x, cell_y, seed_offset);
                uint p = pcg_hash(seed);
                // init_rand(&p,seed);


                // Compute the Poisson parameters for the pixel that contains (x,y)

                int2 uv = convert_int2(max(floor(gaussian), 0.0f));
                float u;
                u = read_imagef(image, sampler, uv).x;
                // u = 0.5f;

                int u_index = (int) floor(u * (float) MAX_GREY_LEVEL);
                float lambda = lambda_lut[u_index].x;
                float exp_lambda = lambda_lut[u_index].y;
                uint grain_count = rand_poisson(&p, lambda, exp_lambda);
                // this just says: how many grains do we have per cell. If their far apart,
                // it's either 0 grains per cell or 1.


                for(uint k=0; k < grain_count; k++) {
                    //draw the grain centre
                    //changed to double precision to avoid incorrect operations
                    double2 grain_center;
                    // this is for when there are more than one grains per cell, it will shift the grain centers.
                    grain_center.x = cell_corner.x + (double) (cell_size * rand_uniform(&p));
                    grain_center.y = cell_corner.y + (double) (cell_size * rand_uniform(&p));

                    float radius_squared;

                    //draw the grain radius
                    if (grain_sigma > 0.0) {
                        //draw a random Gaussian radius, and convert it to log-normal
                        float radius = fmin(exp(mu + sigma * rand_gaussian(&p)), grain_radius_max);
                        radius_squared = radius * radius;
                    }
                    else {
                        radius_squared = grain_radius_squared;
                    }

                    // test distance
                    double distance_squared = square_distance(grain_center, convert_double2(gaussian));
                    if(distance_squared < (double) radius_squared) {
                        value += 1.0f;
                        point_covered = true;
                        break;
                    }
                }
                // end grain

            }
            // end cell_y
        }
        // end cell_x

        point_covered = false;
    }
    //end monte carlo

    value /= (float) samples;
    // value /= cell_size;

    // value.x = (float) o / 4294967295u;

    // value = read_imagef(image, sampler, p);
    // value = (float)(seed_offset) / 4294967295u;

    write_imagef(output, p, (float4) (value));
    // rgba = (float4) (value, 0, 0, 0);
    // write_imagef(output, p, rgba);
}

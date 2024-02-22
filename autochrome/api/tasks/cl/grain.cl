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
    return (a.x - b.x) * (a.x - b.x) + (a.y - a.y) * (a.y - a.y);
}
    
__kernel void grain(
    read_only image2d_t image,
    write_only image2d_t output,
    // __global float2* lambda_lut,
    const int samples,
    const float grain_sigma,
    const float grain_radius,
    const float blur_sigma
    // const uint seed_offset
)
{

    float4 render_bounds = (float4) (0, 0, 1, 1);
    uint seed_offset = 0;

    sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

    int2 output_size = get_image_dim(output);

    int2 p;
    p.x = get_global_id(0);
    p.y = get_global_id(1);

    float ag = 1 / ceil(1 / grain_radius);

    float sX = (float)(output_size.x - 1) / (render_bounds.z - render_bounds.x);
    float sY = (float)(output_size.y - 1) / (render_bounds.w - render_bounds.y);



    float grain_radius_squared = grain_radius * grain_radius;
    float mu, sigma;
    float grain_radius_max = grain_radius;

    //calculate the mu and sigma for the lognormal distribution
    if (grain_sigma > 0.0)
    {
        sigma = sqrt(log((grain_sigma / grain_radius) * (grain_sigma / grain_radius) + 1.0f));
        mu = log(grain_radius) - (sigma * sigma) / 2.0f;
        float normal_quantile = 3.0902; //standard normal quantile for alpha=0.999
        float log_normal_quantile = exp(mu + sigma * normal_quantile);
        grain_radius_max = log_normal_quantile;
    }



    if(p.x < output_size.x && p.y < output_size.y)
    {
        float value = 0.0f;

        //conversion from output grid (xOut,yOut) to input grid (xIn,yIn)
        //we inspect the middle of the output pixel (1/2)
        //the size of a pixel is (render_bounds.z - render_bounds.x) / output_size.x

        float2 render_p;
        render_p.x = render_bounds.x + ((float) p.x + 0.5f) * ((render_bounds.z - render_bounds.x) / (float) output_size.x);
        render_p.y = render_bounds.y + ((float) p.y + 0.5f) * ((render_bounds.w - render_bounds.y) / (float) output_size.y);

        // Simulate Poisson process on the 4 neighborhood cells of (x,y)
        uint p_monte_carlo;
        init_rand(&p_monte_carlo, (uint) 2016 * seed_offset);

        for (int i = 0; i < samples; i++)
        {
            float2 gaussian;
            gaussian.x = render_p.x + blur_sigma * rand_gaussian(&p_monte_carlo) / sX;
            gaussian.y = render_p.y + blur_sigma * rand_gaussian(&p_monte_carlo) / sY;

            // Determine the bounding boxes around the current shifted pixel
            // these operations are set to double precision because the number of cells can be quite large
            uint4 bounds;
            bounds.x = (uint) floor(((double) gaussian.x - (double) grain_radius_max) / (double) ag);
            bounds.y = (uint) floor(((double) gaussian.y - (double) grain_radius_max) / (double) ag);
            bounds.z = (uint) floor(((double) gaussian.x + (double) grain_radius_max) / (double) ag);
            bounds.w = (uint) floor(((double) gaussian.y + (double) grain_radius_max) / (double) ag);
  
            bool point_covered = false; // used to break all for loops





            for(uint cell_x = bounds.x; cell_x <= bounds.z; cell_x++) // x-cell number
            {
                if(point_covered == true)
                {
                    break;
                }

                for(uint cell_y = bounds.y; cell_y <= bounds.w; cell_y++) // y-cell number
                {
                    if(point_covered == true)
                    {
                        break;
                    }

                    double2 cell_corner = (double2) (cell_x, cell_y) * ag;

                    uint seed = cell_seed(cell_x, cell_y, seed_offset);

                    uint p;
                    init_rand(&p,seed);


                    // Compute the Poisson parameters for the pixel that contains (x,y)

                    int2 uv = convert_int2(max(floor(gaussian), 0.0f));
                    float u = read_imagef(image, sampler, uv).x;
                    int u_index = (int) floor(u * (float) MAX_GREY_LEVEL);
                    // float lambda = lambda_lut[u_index].x;
                    // float exp_lambda = lambda_lut[u_index].y;
                    float lambda = 0.05f;
                    float exp_lambda = 0.9f;

                    uint grain_count = rand_poisson(&p, lambda, exp_lambda);

                    for(uint k=0; k < grain_count; k++)
                    {
                        //draw the grain centre
                        //changed to double precision to avoid incorrect operations
                        double2 grain_center = cell_corner;
                        grain_center.x += (double) (ag * rand_uniform(&p));
                        grain_center.y += (double) (ag * rand_uniform(&p));

                        float radius_squared;

                        //draw the grain radius
                        if (grain_sigma > 0.0)
                        {
                            //draw a random Gaussian radius, and convert it to log-normal
                            float radius = fmin(exp(mu + sigma * rand_gaussian(&p)), grain_radius_max);
                            radius_squared = radius * radius;
                        }
                        else
                            radius_squared = grain_radius_squared;

                        // test distance
                        double distance_squared = square_distance(grain_center, convert_double2(gaussian));
                        if(distance_squared < (double) radius_squared)
                        {
                            // value = value + 1.0f;
                            value = value + 1.0f;
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

        value = value / ((float) samples);
        write_imagef(output, p, (float4) (value, 0, 0, 0));
    }
}

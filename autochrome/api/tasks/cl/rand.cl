// https://github.com/alasdairnewson/film_grain_rendering_gpu/blob/master/src/film_grain_rendering.cu

/*
* From http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
* Same strategy as in Gabor noise by example
* Apply hashtable to create cellseed
* Use a linear congruential generator as fast PRNG
*/

/*
* @brief Produce random seed
*
* @param input seed
* @return output, modified seed
*/

uint wang_hash(
	uint seed
    )
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 668265261u;
    seed = seed ^ (seed >> 15u);
    return seed;
}


uint pcg_hash(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/*
* @brief Generate unique seed for a a cell, given the coordinates of the cell
*
* @param (x,y) : input coordinates of the cell
* @param (x,y) : constant offset to change seeds for cells
* @return seed for this cell
*/
uint cell_seed(
    const uint x,
    const uint y,
    const uint offset
    )
{
    const uint period = 65536u; // 2^16 = 65536
    uint seed = (( y % period) * period + (x % period)) + offset;
    if (seed == 0u) seed = 1u;
    return seed;
}

/*
* @brief Initialise internal state of pseudo-random number generator
*
* @param : pointer to the state of the pseudo-random number generator
* @param : seed for initialisation
* @return : void
*/
void init_rand(uint  *state, const uint seed)
{
    uint local_seed = seed;
    *state = pcg_hash(local_seed);
}

/*
* @brief Produce a pseudo-random number and increment the internal state of the pseudo-random number generator
*
* @param : pointer to the state of the pseudo-random number generator
* @return : random integer from 0 to max_unsigned_int (4294967295)
*/

uint rand(uint *state)
{
    // linear congruential generator: procudes correlated output. Similar patterns are visible
    // p.state = 1664525u * p.state + 1013904223u;
    // Xorshift algorithm from George Marsaglia's paper
    *state ^= (*state << 13u);
    *state ^= (*state >> 17u);
    *state ^= (*state << 5u);
    return *state;
}

/*
* @brief Produce uniform random number in the interval [0;1]
*
* @param : pointer to the state of the pseudo-random number generator
* @return : random floating point number in the interval [0;1]
*/
float rand_uniform(uint *state)
{
    return (float) rand(state) / (float) 4294967295u;
}

/*
* @brief Produce random number following a standard normal distribution
*
* @param : pointer to the state of the pseudo-random number generator
* @return : random number following a standard normal distribution
*/
float rand_gaussian(uint  *state)
{
    // Box-Muller method for generating standard Gaussian variate
    float u = rand_uniform(state);
    float v = rand_uniform(state);
    return sqrt(-2.0f * log(u)) * cos(2.0f * M_PI_F * v);
}

/*
* @brief Produce pair of random numbers following a standard normal distribution
*
* @param : pointer to the state of the pseudo-random number generator
* @return : pair of random numbers following a standard normal distribution
*/
float2 rand_gaussian_float2(uint  *state)
{
    // Box-Muller method for generating standard Gaussian variate
    float u = rand_uniform(state);
    float v = rand_uniform(state);

	float2 rand_output;
	rand_output.x = sqrt(-2.0f * log(u)) * cos(2.0f * M_PI_F * v);
	rand_output.y = sqrt(-2.0f * log(u)) * sin(2.0f * M_PI_F * v);
    return(rand_output);
}


/*
* @brief Produce a random number following a Poisson distribution
*
* P(X=k) = l^k * e^-l  / k!
*
* @param : pointer to the state of the pseudo-random number generator
* @param : lambda, parameter of the Poisson distribution
* @param : optional value so that exp(-lambda) need not be recalculated each time we call the Poisson random number generator
* @return : random number following a poisson distribution
*/
uint rand_poisson(uint *state, float lambda, float exp_lambda)
{
	// Inverse transform sampling
	float u=rand_uniform(state);
	uint x = 0u;
	//float prod = exp(-lambda); // this should be passed as an argument if used extensively with the same value lambda
	float prod = exp_lambda;
	float sum = prod;
	while ((u > sum) && (x < floor(10000.0f * lambda))) {
		x = x + 1u;
		prod = prod * lambda / (float) x;
		sum = sum + prod;
	}
	return x;
}

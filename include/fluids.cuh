
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <helper_cuda.h>
#include <Controller.hpp>

extern float3 *h_velocity[2];
extern size_t h_velocity_pitch[2];
extern float *h_pressure[2];
extern size_t h_pressure_pitch[2];
extern float *h_divergence;
extern size_t h_divergence_pitch;
extern float3 *h_dye[2];
extern size_t h_dye_pitch[2];

__global__ void boundary_advect(float **result, float **field, float scale, int width, int height);


// interior_advect.fs


/**
 * @brief Advection Fragment Program
 * 
 * @param result result
 * @param timestep 
 * @param rdx reciprocal of the grid scale x
 * @param x Qty to be advected
 * @param u Velocity profile
 * @param width array width
 * @param height array height
 * @return __global__ 
 */
__global__ void advect(float **result, float timestep, float rdx, float **x, float **u, int width, int height);

// solve_jacobi.fs

/**
 * @brief The Jacobi Iteration Fragment Program Used to Solve Poisson Equations
 * 
 * @param result result
 * @param x x vector (Ax = b)
 * @param b b vector (Ax = b)
 * @param alpha 
 * @param rBeta reciprocal beta
 * @param width array width
 * @param height array height
 */
__global__ void jacobi_float3_kernel(float3 *result, size_t result_pitch,
    float3 *x, size_t x_pitch,
    float3 *b, size_t b_pitch,
    float alpha, float rBeta, int width, int height);
__global__ void jacobi_float_kernel(float *result, size_t result_pitch,
    float *x, size_t x_pitch,
    float *b, size_t b_pitch,
    float alpha, float rBeta, int width, int height);

// divergence.fs

/**
 * @brief The Divergence Fragment Program
 * 
 * @param result divergence
 * @param halfrdx 0.5 / gridscale
 * @param wX vector field (x component)
 * @param wY vector field (y component)
 * @param width array width
 * @param height array height
 * @return __global__ 
 */
__global__ void divergence(float **result, float halfrdx, float **wX, float **wY, int width, int height);

__global__ void float3_to_uint8(uint8_t *result, float3 *input, size_t pitch, int width, int height);

void free_fluids();
void setup_fluids(int width, int height);
void update_fluids(Controller *controller, double timestep);
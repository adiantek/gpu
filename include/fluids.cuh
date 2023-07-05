
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <helper_cuda.h>

#include <Controller.hpp>

extern float2 *h_velocity[2];
extern size_t h_velocity_pitch[2];
extern float *h_pressure[2];
extern size_t h_pressure_pitch[2];
extern float *h_divergence;
extern size_t h_divergence_pitch;
extern float3 *h_dye[2];
extern size_t h_dye_pitch[2];

template <typename T>
__global__ void boundary_advect_kernel(T *result, size_t result_pitch,
                                       T *field, size_t field_pitch,
                                       float scale, int width, int height);

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
template <typename T>
__global__ void advect_kernel(T *result, size_t result_pitch,
                              float timestep, float rdx,
                              T *x, size_t x_pitch,
                              float2 *u, size_t u_pitch,
                              int width, int height);

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
template <typename T>
void jacobi_float2_kernel(T *result, size_t result_pitch,
                          T *x, size_t x_pitch,
                          T *b, size_t b_pitch,
                          float alpha, float rBeta, int width, int height);

// divergence.fs

/**
 * @brief The Divergence Fragment Program
 *
 * @param result divergence
 * @param halfrdx 0.5 / gridscale
 * @param w vector field
 * @param width array width
 * @param height array height
 * @return __global__
 */
__global__ void divergence_kernel(float *result, size_t result_pitch,
                                  float halfrdx,
                                  float2 *w, size_t w_pitch,
                                  int width, int height);

__global__ void float3_to_uint8(uint8_t *result, float3 *input, size_t pitch, int width, int height);

void free_fluids();
void setup_fluids(int width, int height);
void update_fluids(Controller *controller, double timestep);
#include <fluids.cuh>
#include <helper_math.h>

__device__ float3 *m_velocity[2];
__device__ size_t m_velocity_pitch[2];
__device__ float *m_pressure[2];
__device__ size_t m_pressure_pitch[2];
__device__ float *m_divergence;
__device__ size_t m_divergence_pitch;
__device__ float3 *m_dye[2];
__device__ size_t m_dye_pitch[2];

float3 *h_velocity[2];
size_t h_velocity_pitch[2];
float *h_pressure[2];
size_t h_pressure_pitch[2];
float *h_divergence;
size_t h_divergence_pitch;
float3 *h_dye[2];
size_t h_dye_pitch[2];

__global__ void boundary_advect(float **result, float **field, float scale, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0) { // left
        i = 1;
    }
    if (i >= width - 1) { // right
        i = width - 2;
    }
    if (j == 0) { // bottom
        j = 1;
    }
    if (j >= height - 1) { // top
        j = height - 2;
    }
    result[j][i] = field[j][i] * scale;
}

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
__global__ void advect(float **result, float timestep, float rdx, float **x, float **u, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;
    if (i == 0 || j == 0) return;

    float posX = i - timestep * rdx * u[j][i];
    float posY = j - timestep * rdx * u[j][i];

    int i0 = floor(posX);
    int i1 = ceil(posX);
    int j0 = floor(posY);
    int j1 = ceil(posY);

    if (i0 < 0) i0 = 0;
    if (i0 >= width) i0 = width - 1;

    if (i1 < 0) i1 = 0;
    if (i1 >= width) i1 = width - 1;

    if (j0 < 0) j0 = 0;
    if (j0 >= height) j0 = height - 1;

    if (j1 < 0) j1 = 0;
    if (j1 >= height) j1 = height - 1;

    float x00 = x[j0][i0];
    float x01 = x[j0][i1];
    float x10 = x[j1][i0];
    float x11 = x[j1][i1];

    float dx = posX - i0;
    float dy = posY - j0;

    float x0 = x00 * (1 - dx) + x01 * dx;
    float x1 = x10 * (1 - dx) + x11 * dx;

    result[j][i] = x0 * (1 - dy) + x1 * dy;
}


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
__global__ void jacobi(float **result, float **x, float **b, float alpha, float rBeta, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;
    if (i == 0 || j == 0) return;

    // left, right, bottom, and top x samples
    float xL = x[j][i - 1];
    float xR = x[j][i + 1];
    float xB = x[j - 1][i];
    float xT = x[j + 1][i];

    // b sample, from center
    float bC = b[j][i];

    // evaluate Jacobi iteration
    float xNew = (xL + xR + xB + xT + alpha * bC) * rBeta;

    result[j][i] = xNew;
}


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
__global__ void divergence(float **result, float halfrdx, float **wX, float **wY, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;
    if (i == 0 || j == 0) return;

    float wL = wX[j][i - 1];
    float wR = wX[j][i + 1];
    float wB = wY[j - 1][i];
    float wT = wY[j + 1][i];

    float div = halfrdx * ((wR - wL) + (wT - wB));

    result[j][i] = div;
}

__global__ void float3_to_uint8(uint8_t *result, float3 *input, size_t pitch, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    float3 val = input[j * pitch / sizeof(float3) + i];
    j = height - j - 1; // OpenGL flip-y

    result[j * width * 4 + i * 4 + 0] = (uint8_t)(val.x * 255);
    result[j * width * 4 + i * 4 + 1] = (uint8_t)(val.y * 255);
    result[j * width * 4 + i * 4 + 2] = (uint8_t)(val.y * 255);
    result[j * width * 4 + i * 4 + 3] = 255;
}

void free_fluids() {
    checkCudaErrors(cudaFree(h_velocity[0]));
    checkCudaErrors(cudaFree(h_velocity[1]));
    checkCudaErrors(cudaFree(h_pressure[0]));
    checkCudaErrors(cudaFree(h_pressure[1]));
    checkCudaErrors(cudaFree(h_divergence));
    checkCudaErrors(cudaFree(h_dye[0]));
    checkCudaErrors(cudaFree(h_dye[1]));
}

__global__ void setup_dye_texture(int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width || y >= height) {
        return;
    }
    float2 pos = make_float2(x, y);
    float2 center = make_float2(width / 2, height / 2);
    float color = length(pos - center) / 10;
    color = sin(color);
    color = min(color, 1.0f);

    float3 c = make_float3(color, color, color);

    m_dye[0][y * m_dye_pitch[0] / sizeof(float3) + x] = c;
    m_dye[1][y * m_dye_pitch[1] / sizeof(float3) + x] = c; 
}

void setup_fluids(int width, int height) {
    checkCudaErrors(cudaMallocPitch(&h_velocity[0], &h_velocity_pitch[0], width * sizeof(float3), height));
    checkCudaErrors(cudaMemset2D(h_velocity[0], h_velocity_pitch[0], 0, width * sizeof(float3), height));
    checkCudaErrors(cudaMallocPitch(&h_velocity[1], &h_velocity_pitch[1], width * sizeof(float3), height));
    checkCudaErrors(cudaMemset2D(h_velocity[1], h_velocity_pitch[1], 0, width * sizeof(float3), height));

    checkCudaErrors(cudaMallocPitch(&h_pressure[0], &h_pressure_pitch[0], width * sizeof(float), height));
    checkCudaErrors(cudaMemset2D(h_pressure[0], h_pressure_pitch[0], 0, width * sizeof(float), height));
    checkCudaErrors(cudaMallocPitch(&h_pressure[1], &h_pressure_pitch[1], width * sizeof(float), height));
    checkCudaErrors(cudaMemset2D(h_pressure[1], h_pressure_pitch[1], 0, width * sizeof(float), height));

    checkCudaErrors(cudaMallocPitch(&h_divergence, &h_divergence_pitch, width * sizeof(float), height));
    checkCudaErrors(cudaMemset2D(h_divergence, h_divergence_pitch, 0, width * sizeof(float), height));

    checkCudaErrors(cudaMallocPitch(&h_dye[0], &h_dye_pitch[0], width * sizeof(float3), height));
    checkCudaErrors(cudaMemset2D(h_dye[0], h_dye_pitch[0], 0, width * sizeof(float3), height));
    checkCudaErrors(cudaMallocPitch(&h_dye[1], &h_dye_pitch[1], width * sizeof(float3), height));
    checkCudaErrors(cudaMemset2D(h_dye[1], h_dye_pitch[1], 0, width * sizeof(float3), height));

    checkCudaErrors(cudaMemcpyToSymbol(m_velocity, &h_velocity, sizeof(h_velocity)));
    checkCudaErrors(cudaMemcpyToSymbol(m_velocity_pitch, &h_velocity_pitch, sizeof(h_velocity_pitch)));
    checkCudaErrors(cudaMemcpyToSymbol(m_pressure, &h_pressure, sizeof(h_pressure)));
    checkCudaErrors(cudaMemcpyToSymbol(m_pressure_pitch, &h_pressure_pitch, sizeof(h_pressure_pitch)));
    checkCudaErrors(cudaMemcpyToSymbol(m_divergence, &h_divergence, sizeof(h_divergence)));
    checkCudaErrors(cudaMemcpyToSymbol(m_divergence_pitch, &h_divergence_pitch, sizeof(h_divergence_pitch)));
    checkCudaErrors(cudaMemcpyToSymbol(m_dye, &h_dye, sizeof(h_dye)));
    checkCudaErrors(cudaMemcpyToSymbol(m_dye_pitch, &h_dye_pitch, sizeof(h_dye_pitch)));
    
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);
    setup_dye_texture<<<gridDim, blockDim>>>(width, height);
}

void advect_velocity(double timestep) {

}

void advect_dye(double timestep) {
    
}

void update_fluids(double timestep, Controller *controller) {
    timestep = 0.1;

    advect_velocity(timestep);
    advect_dye(timestamp);
}
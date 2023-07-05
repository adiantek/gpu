#include <helper_math.h>

#include <fluids.cuh>

__device__ float2 *m_velocity[2];
__device__ size_t m_velocity_pitch[2];
__device__ float *m_pressure[2];
__device__ size_t m_pressure_pitch[2];
__device__ float *m_divergence;
__device__ size_t m_divergence_pitch;
__device__ float3 *m_dye[2];
__device__ size_t m_dye_pitch[2];

float2 *h_velocity[2];
size_t h_velocity_pitch[2];
float *h_pressure[2];
size_t h_pressure_pitch[2];
float *h_divergence;
size_t h_divergence_pitch;
float3 *h_dye[2];
size_t h_dye_pitch[2];

template <typename T>
__global__ void boundary_advect_kernel(T *result, size_t result_pitch,
                                       T *field, size_t field_pitch,
                                       float scale, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int k = i;
    int l = j;

    if (k == 0) {  // left
        k = 1;
    }
    if (k >= width - 1) {  // right
        k = width - 2;
    }
    if (l == 0) {  // bottom
        l = 1;
    }
    if (l >= height - 1) {  // top
        l = height - 2;
    }
    if (k == i && l == j) {
        scale = 1.0f;
    }
    result[j * result_pitch / sizeof(T) + i] = field[l * field_pitch / sizeof(T) + k] * scale;
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
template <typename T>
__global__ void advect_kernel(T *result, size_t result_pitch,
                              float timestep, float rdx,
                              T *x, size_t x_pitch,
                              float2 *u, size_t u_pitch,
                              int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;
    if (i == 0 || j == 0 || i == width - 1 || j == height - 1) {
        result[j * result_pitch / sizeof(T) + i] = x[j * x_pitch / sizeof(T) + i];
        return;
    }
    float2 pos = make_float2(i, j) - timestep * rdx * u[j * u_pitch / sizeof(float2) + i];

    float2 floorPos = floorf(pos);
    float2 fracPos = pos - floorPos;

    int2 iFloorPos = make_int2(floorPos);

    if (iFloorPos.x < 0) iFloorPos.x = 0;
    if (iFloorPos.x >= width) iFloorPos.x = width - 1;
    if (iFloorPos.y < 0) iFloorPos.y = 0;
    if (iFloorPos.y >= height) iFloorPos.y = height - 1;

    T x00 = x[iFloorPos.y * x_pitch / sizeof(T) + iFloorPos.x];
    T x01 = x[iFloorPos.y * x_pitch / sizeof(T) + iFloorPos.x + 1];
    T x10 = x[(iFloorPos.y + 1) * x_pitch / sizeof(T) + iFloorPos.x];
    T x11 = x[(iFloorPos.y + 1) * x_pitch / sizeof(T) + iFloorPos.x + 1];

    T x0 = x00 * (1 - fracPos.x) + x01 * fracPos.x;
    T x1 = x10 * (1 - fracPos.x) + x11 * fracPos.x;
    T res = x0 * (1 - fracPos.y) + x1 * fracPos.y;

    result[j * result_pitch / sizeof(T) + i] = res;
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
template <typename T>
__global__ void jacobi_kernel(T *result, size_t result_pitch,
                              T *x, size_t x_pitch,
                              T *b, size_t b_pitch,
                              float alpha, float rBeta, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    if (i == 0 || j == 0 || i == width - 1 || j == height - 1) {
        result[j * result_pitch / sizeof(T) + i] = x[j * x_pitch / sizeof(T) + i];
        return;
    }

    // left, right, bottom, and top x samples
    T xL = x[j * x_pitch / sizeof(T) + i - 1];
    T xR = x[j * x_pitch / sizeof(T) + i + 1];
    T xB = x[(j - 1) * x_pitch / sizeof(T) + i];
    T xT = x[(j + 1) * x_pitch / sizeof(T) + i];

    // b sample, from center
    T bC = b[j * b_pitch / sizeof(T) + i];

    // evaluate Jacobi iteration
    T xNew = (xL + xR + xB + xT + alpha * bC) * rBeta;

    result[j * result_pitch / sizeof(T) + i] = xNew;
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
__global__ void divergence_kernel(float *result, size_t result_pitch,
                                  float halfrdx,
                                  float2 *w, size_t w_pitch,
                                  int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;
    if (i == 0 || j == 0 || i == width - 1 || j == height - 1) {
        result[j * result_pitch / sizeof(float) + i] = w[j * w_pitch / sizeof(float2) + i].x;
        return;
    }

    float wL = w[j * w_pitch / sizeof(float2) + i - 1].x;
    float wR = w[j * w_pitch / sizeof(float2) + i + 1].x;
    float wB = w[(j - 1) * w_pitch / sizeof(float2) + i].y;
    float wT = w[(j + 1) * w_pitch / sizeof(float2) + i].y;

    float div = halfrdx * ((wR - wL) + (wT - wB));

    result[j * result_pitch / sizeof(float) + i] = div;
}

__global__ void float3_to_uint8(uint8_t *result, float3 *input, size_t pitch, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    float3 val = input[j * pitch / sizeof(float3) + i];
    j = height - j - 1;  // OpenGL flip-y

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
    checkCudaErrors(cudaMallocPitch(&h_velocity[0], &h_velocity_pitch[0], width * sizeof(float2), height));
    checkCudaErrors(cudaMemset2D(h_velocity[0], h_velocity_pitch[0], 0, width * sizeof(float2), height));
    checkCudaErrors(cudaMallocPitch(&h_velocity[1], &h_velocity_pitch[1], width * sizeof(float2), height));
    checkCudaErrors(cudaMemset2D(h_velocity[1], h_velocity_pitch[1], 0, width * sizeof(float2), height));

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

void advect_velocity(Controller *controller, double timestep) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);
    // 0 -> 1
    boundary_advect_kernel<float2><<<gridDim, blockDim>>>(
        h_velocity[1], h_velocity_pitch[1],
        h_velocity[0], h_velocity_pitch[0],
        -1.0f, width, height);
    // 1 -> 0
    float rdx = 1.0f / width;
    advect_kernel<float2><<<gridDim, blockDim>>>(
        h_velocity[0], h_velocity_pitch[0],
        timestep, rdx,
        h_velocity[1], h_velocity_pitch[1],
        h_velocity[1], h_velocity_pitch[1],
        width, height);
}

void advect_dye(Controller *controller, double timestep) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);
    // 0 -> 1
    boundary_advect_kernel<float3><<<gridDim, blockDim>>>(
        h_dye[1], h_dye_pitch[1],
        h_dye[0], h_dye_pitch[0],
        -1.0f, width, height);
    // 1 -> 0
    float rdx = 1.0f / width;
    advect_kernel<float3><<<gridDim, blockDim>>>(
        h_dye[0], h_dye_pitch[0],
        timestep, rdx,
        h_dye[1], h_dye_pitch[1],
        h_velocity[1], h_velocity_pitch[1],
        width, height);
}

void apply_force(Controller *controller) {
}

void diffuse_velocity(Controller *controller, double timestep) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);
    const float m_viscosity = 0.0001f;
    float alpha = (2.0f / width) * (2.0f / height) / (m_viscosity * timestep);
    float rBeta = 1.0f / (4 + alpha);
    for (int i = 0; i < 15; i++) {
        // 0 -> 1
        jacobi_kernel<float2><<<gridDim, blockDim>>>(
            h_velocity[1], h_velocity_pitch[1],
            h_velocity[0], h_velocity_pitch[0],
            h_velocity[0], h_velocity_pitch[0],
            alpha, rBeta,
            width, height);
        // 1 -> 0
        jacobi_kernel<float2><<<gridDim, blockDim>>>(
            h_velocity[0], h_velocity_pitch[0],
            h_velocity[1], h_velocity_pitch[1],
            h_velocity[1], h_velocity_pitch[1],
            alpha, rBeta,
            width, height);
    }
}

void divergence(Controller *controller) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);
    float halfrdx = 0.5f / width;
    divergence_kernel<<<gridDim, blockDim>>>(
        h_divergence, h_divergence_pitch,
        halfrdx,
        h_velocity[0], h_velocity_pitch[0],
        width, height);
}

void computePressure() {
}

void gradientSubtraction() {
}

void update_fluids(Controller *controller, double timestep) {
    timestep = 0.1;

    advect_velocity(controller, timestep);
    advect_dye(controller, timestep);

    if (controller->mouseButtons[GLFW_MOUSE_BUTTON_LEFT]) {
        apply_force(controller);
    }

    diffuse_velocity(controller, timestep);

    divergence(controller);
    computePressure();
    gradientSubtraction();
}
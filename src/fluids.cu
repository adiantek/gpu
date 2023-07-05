#include <Logger.h>
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

__global__ void swap_velocity() {
    float2 *tmp_velocity = m_velocity[0];
    m_velocity[0] = m_velocity[1];
    m_velocity[1] = tmp_velocity;

    size_t tmp_velocity_pitch = m_velocity_pitch[0];
    m_velocity_pitch[0] = m_velocity_pitch[1];
    m_velocity_pitch[1] = tmp_velocity_pitch;
}

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
    float2 fractPos = pos - floorPos;

    int2 iFloorPos = make_int2(floorPos);

    if (iFloorPos.x < 0) {
        iFloorPos.x = 0;
        fractPos.x = 0;
    }
    if (iFloorPos.x >= width - 1) {
        iFloorPos.x = width - 2;
        fractPos.x = 1;
    }
    if (iFloorPos.y < 0) {
        iFloorPos.y = 0;
        fractPos.y = 0;
    }
    if (iFloorPos.y >= height - 1) {
        iFloorPos.y = height - 2;
        fractPos.y = 1;
    }

    T x00 = x[iFloorPos.y * x_pitch / sizeof(T) + iFloorPos.x];
    T x01 = x[iFloorPos.y * x_pitch / sizeof(T) + iFloorPos.x + 1];
    T x10 = x[(iFloorPos.y + 1) * x_pitch / sizeof(T) + iFloorPos.x];
    T x11 = x[(iFloorPos.y + 1) * x_pitch / sizeof(T) + iFloorPos.x + 1];

    T x0 = x00 * (1 - fractPos.x) + x01 * fractPos.x;
    T x1 = x10 * (1 - fractPos.x) + x11 * fractPos.x;
    T res = x0 * (1 - fractPos.y) + x1 * fractPos.y;

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

__global__ void apply_force_kernel(float2 *result, size_t result_pitch,
                                   float2 *u, size_t u_pitch,
                                   float radius, float2 point, float2 F,
                                   int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    if (i == 0 || j == 0 || i == width - 1 || j == height - 1) {
        result[j * result_pitch / sizeof(float2) + i] = u[j * u_pitch / sizeof(float2) + i];
        return;
    }

    float2 uv = u[j * u_pitch / sizeof(float2) + i];
    float2 pos = make_float2(i, j);
    float dist = length(pos - point);
    float2 v_xy = F * expf(-(dist * dist) / radius);
    result[j * result_pitch / sizeof(float2) + i] = uv + v_xy;
}

__global__ void float3_to_uint8(uint8_t *result, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    float2 val_vel = m_velocity[0][j * m_velocity_pitch[0] / sizeof(float2) + i];
    float val_pressure = m_pressure[0][j * m_pressure_pitch[0] / sizeof(float) + i];
    float val_divergence = m_divergence[j * m_divergence_pitch / sizeof(float) + i];
    float3 val_dye = m_dye[0][j * m_dye_pitch[0] / sizeof(float3) + i];

    val_vel = clamp(val_vel, -1.0f, 1.0f) * 0.5f + 0.5f;
    val_pressure = clamp(val_pressure, -1.0f, 1.0f) * 0.5f + 0.5f;
    val_divergence = clamp(val_divergence, -1.0f, 1.0f) * 0.5f + 0.5f;
    val_dye = clamp(val_dye, 0.0f, 1.0f);

    width *= 2;
    height *= 2;

    j = height - j - 1;  // OpenGL flip-y

    result[j * width * 4 + i * 4 + 0] = (uint8_t)(val_dye.x * 255);
    result[j * width * 4 + i * 4 + 1] = (uint8_t)(val_dye.y * 255);
    result[j * width * 4 + i * 4 + 2] = (uint8_t)(val_dye.z * 255);
    result[j * width * 4 + i * 4 + 3] = 255;

    i += width / 2;

    result[j * width * 4 + i * 4 + 0] = (uint8_t)(val_vel.x * 255);
    result[j * width * 4 + i * 4 + 1] = (uint8_t)(val_vel.y * 255);
    result[j * width * 4 + i * 4 + 2] = 255;
    result[j * width * 4 + i * 4 + 3] = 255;

    j -= height / 2;

    result[j * width * 4 + i * 4 + 0] = (uint8_t)(val_divergence * 255);
    result[j * width * 4 + i * 4 + 1] = 255;
    result[j * width * 4 + i * 4 + 2] = 255;
    result[j * width * 4 + i * 4 + 3] = 255;

    i -= width / 2;

    result[j * width * 4 + i * 4 + 0] = (uint8_t)(val_pressure * 255);
    result[j * width * 4 + i * 4 + 1] = 255;
    result[j * width * 4 + i * 4 + 2] = 255;
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

uint8_t *read_fully_file(const char *name, size_t *size) {
    FILE *f = fopen(name, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file %s\n", name);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(*size);
    if (!data) {
        fprintf(stderr, "Failed to allocate %zu bytes\n", *size);
        exit(1);
    }
    if (fread(data, 1, *size, f) != *size) {
        fprintf(stderr, "Failed to read %zu bytes\n", *size);
        exit(1);
    }
    fclose(f);
    return data;
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

    size_t m_texture_size;
    uint8_t *m_texture = read_fully_file("winxp.raw", &m_texture_size);
    float3 *m_texture_float = (float3 *)malloc(h_dye_pitch[0] * height);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int sx = x * 512 / width;
            int sy = y * 512 / height;
            int index = sy * 512 + sx;
            m_texture_float[y * h_dye_pitch[0] / sizeof(float3) + x] = make_float3(
                m_texture[index * 3 + 0] / 255.0f,
                m_texture[index * 3 + 1] / 255.0f,
                m_texture[index * 3 + 2] / 255.0f);
        }
    }
    free(m_texture);
    cudaMemcpy(h_dye[0], m_texture_float, h_dye_pitch[0] * height, cudaMemcpyHostToDevice);
    cudaMemcpy(h_dye[1], m_texture_float, h_dye_pitch[1] * height, cudaMemcpyHostToDevice);
    free(m_texture_float);
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
        0.0f, width, height);
    // 1 -> 0
    float rdx = 1.0f / (2.0f / width);
    advect_kernel<float3><<<gridDim, blockDim>>>(
        h_dye[0], h_dye_pitch[0],
        timestep, rdx,
        h_dye[1], h_dye_pitch[1],
        h_velocity[0], h_velocity_pitch[0],
        width, height);
}

void apply_force(Controller *controller) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);
    float radius = sqrtf(controller->deltaX * controller->deltaX + controller->deltaY * controller->deltaY);
    apply_force_kernel<<<gridDim, blockDim>>>(
        h_velocity[1], h_velocity_pitch[1],
        h_velocity[0], h_velocity_pitch[0],
        radius,
        make_float2(controller->mouseX, controller->mouseY),
        make_float2(controller->deltaX, controller->deltaY),
        width, height);

    float2 *tmp = h_velocity[0];
    h_velocity[0] = h_velocity[1];
    h_velocity[1] = tmp;

    size_t tmp_pitch = h_velocity_pitch[0];
    h_velocity_pitch[0] = h_velocity_pitch[1];
    h_velocity_pitch[1] = tmp_pitch;

    swap_velocity<<<1, 1>>>();
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
    float halfrdx = 0.5f / (2.0f / width);
    divergence_kernel<<<gridDim, blockDim>>>(
        h_divergence, h_divergence_pitch,
        halfrdx,
        h_velocity[0], h_velocity_pitch[0],
        width, height);
}

void computePressure(Controller *controller) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);

    // cudaMemset2D(h_pressure[0], h_pressure_pitch[0], 0, width * sizeof(float), height);
    // cudaMemset2D(h_pressure[1], h_pressure_pitch[1], 0, width * sizeof(float), height);

    for (int i = 0; i < 70; i++) {
        boundary_advect_kernel<float><<<gridDim, blockDim>>>(
            h_pressure[1], h_pressure_pitch[1],
            h_pressure[0], h_pressure_pitch[0],
            1.0f, width, height);

        jacobi_kernel<float><<<gridDim, blockDim>>>(
            h_pressure[0], h_pressure_pitch[0],
            h_pressure[1], h_pressure_pitch[1],
            h_divergence, h_divergence_pitch,
            -(2.0f / width) * (2.0f / height), 0.25f,
            width, height);
    }
}

void gradientSubtraction(Controller *controller) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);

    boundary_advect_kernel<float2><<<gridDim, blockDim>>>(
        h_velocity[1], h_velocity_pitch[1],
        h_velocity[0], h_velocity_pitch[0],
        -1.0f, width, height);

    float2 *tmp = h_velocity[0];
    h_velocity[0] = h_velocity[1];
    h_velocity[1] = tmp;

    size_t tmp_pitch = h_velocity_pitch[0];
    h_velocity_pitch[0] = h_velocity_pitch[1];
    h_velocity_pitch[1] = tmp_pitch;

    swap_velocity<<<1, 1>>>();

    // TODO: subtract gradient
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
    computePressure(controller);
    gradientSubtraction(controller);
}
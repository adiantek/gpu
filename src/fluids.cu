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
void swap_velocity_host() {
    float2 *tmp_velocity = h_velocity[0];
    h_velocity[0] = h_velocity[1];
    h_velocity[1] = tmp_velocity;
    size_t tmp_velocity_pitch = h_velocity_pitch[0];
    h_velocity_pitch[0] = h_velocity_pitch[1];
    h_velocity_pitch[1] = tmp_velocity_pitch;
}
__global__ void swap_dye() {
    float3 *tmp_dye = m_dye[0];
    m_dye[0] = m_dye[1];
    m_dye[1] = tmp_dye;
    size_t tmp_dye_pitch = m_dye_pitch[0];
    m_dye_pitch[0] = m_dye_pitch[1];
    m_dye_pitch[1] = tmp_dye_pitch;
}
void swap_dye_host() {
    float3 *tmp_dye = h_dye[0];
    h_dye[0] = h_dye[1];
    h_dye[1] = tmp_dye;
    size_t tmp_dye_pitch = h_dye_pitch[0];
    h_dye_pitch[0] = h_dye_pitch[1];
    h_dye_pitch[1] = tmp_dye_pitch;
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
                              float dt, float dissipation,
                              T *x, size_t x_pitch,
                              float2 *u, size_t u_pitch,
                              int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;
    float2 pos = make_float2(i, j) - dt * u[j * u_pitch / sizeof(float2) + i];

    float2 floorPos = floorf(pos);
    float2 fractPos = pos - floorPos;

    int2 iFloorPos = make_int2(floorPos);

    if (iFloorPos.x < 0) {
        iFloorPos.x = 0;
        fractPos.x = 0;
    }
    if (iFloorPos.x >= width - 1) {
        iFloorPos.x = width - 1;
        fractPos.x = 1;
    }
    if (iFloorPos.y < 0) {
        iFloorPos.y = 0;
        fractPos.y = 0;
    }
    if (iFloorPos.y >= height - 1) {
        iFloorPos.y = height - 1;
        fractPos.y = 1;
    }

    T x00 = x[iFloorPos.y * x_pitch / sizeof(T) + iFloorPos.x];
    T x01 = x[iFloorPos.y * x_pitch / sizeof(T) + min(iFloorPos.x + 1, width - 1)];
    T x10 = x[min(iFloorPos.y + 1, height - 1) * x_pitch / sizeof(T) + iFloorPos.x];
    T x11 = x[min(iFloorPos.y + 1, height - 1) * x_pitch / sizeof(T) + min(iFloorPos.x + 1, width - 1)];

    T x0 = x00 * (1 - fractPos.x) + x01 * fractPos.x;
    T x1 = x10 * (1 - fractPos.x) + x11 * fractPos.x;
    T res = x0 * (1 - fractPos.y) + x1 * fractPos.y;

    result[j * result_pitch / sizeof(T) + i] = res / (1.0 + dt * dissipation);
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

    float2 C = w[j * w_pitch / sizeof(float2) + i];
    
    float wL = w[j * w_pitch / sizeof(float2) + max(i - 1, 0)].x;
    float wR = w[j * w_pitch / sizeof(float2) + min(i + 1, width - 1)].x;
    float wB = w[max(j - 1, 0) * w_pitch / sizeof(float2) + i].y;
    float wT = w[min(j + 1, height - 1) * w_pitch / sizeof(float2) + i].y;

    if (i == 0) wL = -C.x;
    if (i == width - 1) wR = -C.x;
    if (j == 0) wB = -C.y;
    if (j == height - 1) wT = -C.y;

    float div = halfrdx * ((wR - wL) + (wT - wB));

    result[j * result_pitch / sizeof(float) + i] = div;
}

template <typename T>
__global__ void apply_force_kernel(T *result, size_t result_pitch,
                                   T *u, size_t u_pitch,
                                   float radius, float2 point, T F,
                                   int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    if (i == 0 || j == 0 || i == width - 1 || j == height - 1) {
        result[j * result_pitch / sizeof(T) + i] = u[j * u_pitch / sizeof(T) + i];
        return;
    }

    T uv = u[j * u_pitch / sizeof(T) + i];
    float2 pos = make_float2(i, j);
    float dist = length(pos - point);
    T v_xy = F * expf(-(dist * dist) / radius);
    result[j * result_pitch / sizeof(T) + i] = uv + v_xy;
}


__global__ void gradient_kernel(float2 *result, size_t result_pitch,
                                float halfrdx,
                                float *p, size_t p_pitch,
                                float2 *w, size_t w_pitch,
                                int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    float pL = p[j * p_pitch / sizeof(float) + max(i - 1, 0)];
    float pR = p[j * p_pitch / sizeof(float) + min(i + 1, width - 1)];
    float pB = p[max(j - 1, 0) * p_pitch / sizeof(float) + i];
    float pT = p[min(j + 1, height - 1) * p_pitch / sizeof(float) + i];

    float2 uNew = w[j * w_pitch / sizeof(float2) + i];
    uNew -= halfrdx * make_float2(pR - pL,  pT - pB);

    result[j * result_pitch / sizeof(float2) + i] = uNew;
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

void apply_force(Controller *controller) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);

    float dX = controller->deltaX * 10.0f;
    float dY = controller->deltaY * 10.0f;

    float radius = sqrtf(dX * dX + dY * dY);
    if (radius < 1.0f) {
        return;
    }
    apply_force_kernel<float2><<<gridDim, blockDim>>>(
        h_velocity[1], h_velocity_pitch[1],
        h_velocity[0], h_velocity_pitch[0],
        radius,
        make_float2(controller->mouseX, controller->mouseY),
        make_float2(dX, dY),
        width, height);

    swap_velocity<<<1, 1>>>();
    swap_velocity_host();

    // float3 color = hsv2rgb(rand() * 1.0f / RAND_MAX, 1.0f, 1.0f) * 0.15f;
    // printf("%f %f %f\n", color.x, color.y, color.z);

    apply_force_kernel<float3><<<gridDim, blockDim>>>(
        h_dye[1], h_dye_pitch[1],
        h_dye[0], h_dye_pitch[0],
        radius,
        make_float2(controller->mouseX, controller->mouseY),
        controller->currentColor * 0.01f,
        width, height);

    swap_dye<<<1, 1>>>();
    swap_dye_host();
}

void divergence(Controller *controller) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);
    float halfrdx = 0.5f;
    divergence_kernel<<<gridDim, blockDim>>>(
        h_divergence, h_divergence_pitch,
        halfrdx,
        h_velocity[0], h_velocity_pitch[0],
        width, height);
}

void advect_velocity(Controller *controller, double timestep) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);
    advect_kernel<float2><<<gridDim, blockDim>>>(
        h_velocity[1], h_velocity_pitch[1],
        timestep, 0.2f,
        h_velocity[0], h_velocity_pitch[0],
        h_velocity[0], h_velocity_pitch[0],
        width, height);

    swap_velocity<<<1, 1>>>();
    swap_velocity_host();
}

void advect_dye(Controller *controller, double timestep) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);

    advect_kernel<float3><<<gridDim, blockDim>>>(
        h_dye[1], h_dye_pitch[1],
        timestep, 1.0f,
        h_dye[0], h_dye_pitch[0],
        h_velocity[0], h_velocity_pitch[0],
        width, height);

    swap_dye<<<1, 1>>>();
    swap_dye_host();
}

void computePressure(Controller *controller) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);

    cudaMemset2D(h_pressure[0], h_pressure_pitch[0], 0, width * sizeof(float), height);

    for (int i = 0; i < 5; i++) {
        jacobi_kernel<float><<<gridDim, blockDim>>>(
            h_pressure[1], h_pressure_pitch[1],
            h_pressure[0], h_pressure_pitch[0],
            h_divergence, h_divergence_pitch,
            -1.0f, 0.25f,
            width, height);
        jacobi_kernel<float><<<gridDim, blockDim>>>(
            h_pressure[0], h_pressure_pitch[0],
            h_pressure[1], h_pressure_pitch[1],
            h_divergence, h_divergence_pitch,
            -1.0f, 0.25f,
            width, height);
    }
}

void gradient(Controller *controller) {
    int width = controller->width;
    int height = controller->height;
    dim3 gridDim((width + 31) / 32, (height + 31) / 32);
    dim3 blockDim(32, 32);

    gradient_kernel<<<gridDim, blockDim>>>(
        h_velocity[1], h_velocity_pitch[1],
        1.0f,
        h_pressure[0], h_pressure_pitch[0],
        h_velocity[0], h_velocity_pitch[0],
        width, height);

    swap_velocity<<<1, 1>>>();
    swap_velocity_host();
}

void update_fluids(Controller *controller, double timestep) {
    if (controller->mouseButtons[GLFW_MOUSE_BUTTON_LEFT]) {
        apply_force(controller); // -> velocity
    }

    divergence(controller); // velocity -> divergence
    computePressure(controller); // divergence -> pressure
    gradient(controller); // velocity,pressure -> velocity

    advect_velocity(controller, timestep); // velocity -> velocity
    advect_dye(controller, timestep); // velocity,dye -> dye
}
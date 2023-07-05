#include <0glad.h>  // glad must be included before glfw
#include <GLFW/glfw3.h>
#include <Logger.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <imgui/imgui_single_file.h>
#include <main.h>

#include <Controller.hpp>
#include <fluids.cuh>

GLuint frameBuffer;
GLuint renderBuffer;
struct cudaGraphicsResource *cudaResource;
cudaArray_t cudaResourceArr;
uint8_t *cudaResourcePtr;

__global__ void cudaInit(uint8_t *ptr, int w, int h) {
    for (int i = 0; i < w * h; i++) {
        ptr[i * 4 + 0] = 255;
        ptr[i * 4 + 1] = 0;
        ptr[i * 4 + 2] = 0;
        ptr[i * 4 + 3] = 255;
    }
}

void createRenderBuffer(GLuint *renderBuffer, int width, int height) {
    glGenFramebuffers(1, &frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
    glGenRenderbuffers(1, renderBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, *renderBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width * 2, height * 2);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, *renderBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResource, *renderBuffer, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaResourceArr, cudaResource, 0, 0));

    checkCudaErrors(cudaMalloc(&cudaResourcePtr, width * 2 * height * 2 * 4));
    cudaInit<<<1, 1>>>(cudaResourcePtr, width * 2, height * 2);
    setup_fluids(width, height);
}

void resizeRenderBuffer(int width, int height) {
    checkCudaErrors(cudaFree(cudaResourcePtr));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource, 0));
    checkCudaErrors(cudaGraphicsUnregisterResource(cudaResource));
    glDeleteRenderbuffers(1, &renderBuffer);
    glDeleteFramebuffers(1, &frameBuffer);
    free_fluids();
    createRenderBuffer(&renderBuffer, width, height);
}

void glfwErrorCallback(int, const char *err_str) {
    LOGE("GLFW Error: %s", err_str);
}

void messageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam) {
    LOGE("GL CALLBACK [%u]: %s type = 0x%x, source = 0x%x, severity = 0x%x, message = %s", id, (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, source, severity, message);
}

__global__ void cudaDraw(uint8_t *ptr, int mouseX, int mouseY, int w, int h) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > w || y > h) {
        return;
    }
    int dx = x - mouseX;
    int dy = y - (h - mouseY);
    int d = dx * dx + dy * dy;
    if (d < 255 * 8) {
        uint8_t val = (uint8_t)(255 - d / 8);
        int index = (y * w + x) * 4 + 1;
        if (ptr[index] < val) {
            ptr[index] = val;
        }
    }
}

int main(int argc, char **argv) {
    LOGI("Hello, world!");
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        LOGE("Failed to initialize GLFW.");
        return EXIT_FAILURE;
    }
    LOGI("GLFW initialized.");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(1024, 1024, "Lubię trójkąty", NULL, NULL);
    if (!window) {
        LOGE("Failed to create window.");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    LOGI("Window created.");
    glfwMakeContextCurrent(window);
    if (!gladLoadGL()) {
        LOGE("gladLoadGL() failed");
        return EXIT_FAILURE;
    }
    // disable NVIDIA notification about using VIDEO memory
    glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DEBUG_SEVERITY_NOTIFICATION, 0, 0, GL_FALSE);
    glDebugMessageCallback(messageCallback, 0);
    glEnable(GL_DEBUG_OUTPUT);

    LOGI("GL_VERSION: %s", glGetString(GL_VERSION));
    LOGI("GL_RENDERER: %s", glGetString(GL_RENDERER));
    LOGI("GL_VENDOR: %s", glGetString(GL_VENDOR));
    LOGI("GL_SHADING_LANGUAGE_VERSION: %s", glGetString(GL_SHADING_LANGUAGE_VERSION));

    controller = new Controller(window);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(nullptr);

    // bool show_demo_window = true;

    double currTime = glfwGetTime();
    double prevTime = currTime;

    createRenderBuffer(&renderBuffer, controller->width, controller->height);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        // ImGui::ShowDemoWindow(&show_demo_window);
        ImGui::Render();

        int w = controller->width;
        int h = controller->height;

        currTime = glfwGetTime();
        update_fluids(controller, currTime - prevTime);
        prevTime = currTime;

        // cudaMemcpy(cudaResourcePtr, cudaClearPtr, 1280 * 720 * 4, cudaMemcpyDeviceToDevice);
        dim3 gridDim((w + 31) / 32, (h + 31) / 32);
        dim3 blockDim(32, 32);
        float3_to_uint8<<<gridDim, blockDim>>>(cudaResourcePtr, w, h);
        if (controller->mouseButtons[GLFW_MOUSE_BUTTON_LEFT]) {
            // cudaDraw<<<gridDim, blockDim>>>(cudaResourcePtr, floor(controller->mouseX), floor(controller->mouseY), w, h);
        }
        cudaMemcpy2DToArray(cudaResourceArr, 0, 0, cudaResourcePtr, w * 4 * 2, w * 4 * 2, h * 2, cudaMemcpyDeviceToDevice);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, w * 2, h * 2, 0, 0, w * 2, h * 2, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    delete controller;

    glfwTerminate();
    return 0;
}

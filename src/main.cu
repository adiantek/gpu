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

float3 hsv2rgb(float h, float s, float v) {
	float r, g, b;
	
	int i = floor(h * 6);
	float f = h * 6 - i;
	float p = v * (1 - s);
	float q = v * (1 - f * s);
	float t = v * (1 - (1 - f) * s);
	
	switch (i % 6) {
		case 0: r = v, g = t, b = p; break;
		case 1: r = q, g = v, b = p; break;
		case 2: r = p, g = v, b = t; break;
		case 3: r = p, g = q, b = v; break;
		case 4: r = t, g = p, b = v; break;
		case 5: r = v, g = p, b = q; break;
	}
	
	float3 color;
	color.x = r * 255;
	color.y = g * 255;
	color.z = b * 255;
	
	return color;
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
    GLFWwindow *window = glfwCreateWindow(1024, 1024, "Płyny wysokoprocentowe", NULL, NULL);
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

        int w = controller->width;
        int h = controller->height;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            ImGui::Begin("Plyny");
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Checkbox("Paused", &controller->paused);
            ImGui::SliderInt("Iterations", &controller->iterations, 1, 200);
            ImGui::SliderFloat("Timer", &controller->timer, 0.0f, 5.0f);
            ImGui::SliderFloat("Velocity decay", &controller->velocityDecay, 0.0f, 2.0f);
            ImGui::SliderFloat("Dye decay", &controller->dyeDecay, 0.0f, 2.0f);
            ImGui::SliderFloat("Dye color focus", &controller->dyeColor, 0.0f, 2.0f);
            ImGui::SliderFloat("Radius", &controller->radius, 0.0f, 20.0f);
            ImGui::SliderFloat("Divergence rdx", &controller->divergenceRdx, 0.0f, 1.0f);
            ImGui::SliderFloat("Gradient rdx", &controller->gradientRdx, 0.0f, 1.0f);
            ImGui::SliderAngle("Wind angle", &controller->windAngle, 0.0f, 360.0f);
            ImGui::SliderFloat("Wind strength", &controller->windStrength, 0.0f, 100.0f);
            if (ImGui::Button("Clear velocity")) {
                cudaMemset2D(h_velocity[0], h_velocity_pitch[0], 0, w * sizeof(float2), h);
            }
            if (ImGui::Button("Clear dye")) {
                cudaMemset2D(h_dye[0], h_dye_pitch[0], 0, w * sizeof(float3), h);
            }
            if (ImGui::Button("Load image")) {
                load_image(w, h);
            }
            ImGui::End();
        }
        // ImGui::ShowDemoWindow(&show_demo_window);
        ImGui::Render();

        float hue = (float) (currTime / 5.0);
        hue = hue - floor(hue);
        controller->currentColor = hsv2rgb(hue, 1.0f, 1.0f);

        currTime = glfwGetTime();
        update_fluids(controller, currTime - prevTime);
        prevTime = currTime;

        // cudaMemcpy(cudaResourcePtr, cudaClearPtr, 1280 * 720 * 4, cudaMemcpyDeviceToDevice);
        dim3 gridDim((w + GRID_SIZE - 1) / GRID_SIZE, (h + GRID_SIZE - 1) / GRID_SIZE);
        dim3 blockDim(GRID_SIZE, GRID_SIZE);
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

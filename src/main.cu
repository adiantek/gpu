#include <Controller.hpp>
#include <GLFW/glfw3.h>
#include <Logger.h>
#include <main.h>
#include <imgui/imgui_single_file.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>

void glfwErrorCallback(int, const char *err_str) {
    LOGE("GLFW Error: %s", err_str);
}

void messageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam) {
    LOGE("GL CALLBACK [%u]: %s type = 0x%x, source = 0x%x, severity = 0x%x, message = %s", id, (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, source, severity, message);
}

__global__ void cudaHello() {
    printf("Hello World from GPU!\n");
}

int main(int argc, char **argv) {
    cudaHello<<<1, 1>>>();
    LOGI("Hello, world!");
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        LOGE("Failed to initialize GLFW.");
        return 1;
    }
    LOGI("GLFW initialized.");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(1280, 720, "Lubię trójkąty", NULL, NULL);
    if (!window) {
        LOGE("Failed to create window.");
        glfwTerminate();
        return 1;
    }
    LOGI("Window created.");
    glfwMakeContextCurrent(window);
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

    bool show_demo_window = true;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::ShowDemoWindow(&show_demo_window);
        ImGui::Render();
        glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        

        glfwSwapBuffers(window);
    }

    delete controller;

    glfwTerminate();
    return 0;
}

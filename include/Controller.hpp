#pragma once

#include <0glad.h>
#include <GLFW/glfw3.h>
#include <stdbool.h>
#include <stdint.h>
#include <vector_types.h>

class Controller {
   public:
    Controller(GLFWwindow *window);
    ~Controller();

    GLFWwindow *window;
    int width;
    int height;
    double deltaX;
    double deltaY;
    double mouseX;
    double mouseY;
    float3 currentColor;
    
    bool keys[GLFW_KEY_LAST];
    bool mouseButtons[GLFW_MOUSE_BUTTON_LAST];

    static const int MODE_PASSIVE = 0;
    static const int MODE_ACTIVE = 1;

    static void onWindowSizeChange(GLFWwindow *window, int width, int height);
    static void onKeyPress(GLFWwindow *window, int key, int scancode, int action, int mode);
    static void onMouseButtonPress(GLFWwindow *window, int button, int action, int mode);
    static void onCursorPositionChange(GLFWwindow *window, double xPos, double yPos);

    bool paused = false;
    float velocityDecay = 0.2f;
    float dyeDecay = 1.0f;
    float timer = 1.0f;
    float dyeColor = 1.0f;
    float radius = 20.0f;
    int iterations = 10;
    float divergenceRdx = 0.5f;
    float gradientRdx = 1.0f;
    float windAngle = 0.0f;
    float windStrength = 0.0f;
};

#ifdef __cplusplus
extern "C" {
#endif

extern Controller *controller;

#ifdef __cplusplus
}
#endif
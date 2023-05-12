#pragma once

#include <GLFW/glfw3.h>
#include <stdbool.h>
#include <stdint.h>

class Controller {
   public:
    Controller(GLFWwindow *window);
    ~Controller();

    GLFWwindow *window;
    int width;
    int height;
    double mouseX;
    double mouseY;
    
    bool keys[GLFW_KEY_LAST];
    bool mouseButtons[GLFW_MOUSE_BUTTON_LAST];

    static const int MODE_PASSIVE = 0;
    static const int MODE_ACTIVE = 1;

    static void onWindowSizeChange(GLFWwindow *window, int width, int height);
    static void onKeyPress(GLFWwindow *window, int key, int scancode, int action, int mode);
    static void onMouseButtonPress(GLFWwindow *window, int button, int action, int mode);
    static void onCursorPositionChange(GLFWwindow *window, double xPos, double yPos);

};

#ifdef __cplusplus
extern "C" {
#endif

extern Controller *controller;

#ifdef __cplusplus
}
#endif
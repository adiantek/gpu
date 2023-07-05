#include <Controller.hpp>
#include <Logger.h>
#include <main.h>

Controller *controller = nullptr;

Controller::Controller(GLFWwindow *window) {
    controller = this;

    this->window = window;

    for (int i = 0; i < GLFW_KEY_LAST; i++) {
        this->keys[i] = false;
    }
    for (int i = 0; i < GLFW_MOUSE_BUTTON_LAST; i++) {
        this->mouseButtons[i] = false;
    }
    glfwGetWindowSize(window, &this->width, &this->height);

    glfwSetWindowSizeCallback(window, onWindowSizeChange);
    glfwSetKeyCallback(window, onKeyPress);
    glfwSetMouseButtonCallback(window, onMouseButtonPress);
    glfwSetCursorPosCallback(window, onCursorPositionChange);
}

Controller::~Controller() {
}

void Controller::onWindowSizeChange(GLFWwindow *window, int width, int height) {
    if (width < 1) {
        width = 1;
    }
    if (height < 1) {
        height = 1;
    }
    controller->width = width;
    controller->height = height;

    resizeRenderBuffer(width, height);
}
void Controller::onKeyPress(GLFWwindow *window, int key, int scancode, int action, int mode) {
    if (key >= GLFW_KEY_LAST) {
        return;
    }
    if (action == 1) {
        controller->keys[key] = true;
    } else {
        controller->keys[key] = false;
    }
}
void Controller::onMouseButtonPress(GLFWwindow *window, int button, int action, int mode) {
    if (button >= GLFW_MOUSE_BUTTON_LAST) {
        return;
    }
    if (action == 1) {
        controller->mouseButtons[button] = true;
    } else {
        controller->mouseButtons[button] = false;
    }
}
void Controller::onCursorPositionChange(GLFWwindow *window, double xPos, double yPos) {
    controller->deltaX = xPos - controller->mouseX;
    controller->deltaY = yPos - controller->mouseY;
    controller->mouseX = xPos;
    controller->mouseY = yPos;
}
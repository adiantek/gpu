cmake_minimum_required(VERSION 3.22)
project(gpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
enable_language(CUDA)

message(STATUS "Cmake prefix path: ${CMAKE_PREFIX_PATH}")
include_directories("include")
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.c ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(${PROJECT_NAME} ${SOURCES})

add_compile_definitions(GLM_FORCE_CUDA)
add_compile_definitions(GL_GLEXT_PROTOTYPES)

find_package(glfw3 3.3 REQUIRED)

target_link_libraries(${PROJECT_NAME} "glfw")

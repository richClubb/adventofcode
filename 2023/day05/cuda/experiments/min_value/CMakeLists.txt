cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(min_value LANGUAGES C CXX CUDA)

add_executable(
    ${PROJECT_NAME}
    "${PROJECT_SOURCE_DIR}/main.cu"
)

target_include_directories(
    "${PROJECT_NAME}"
    PRIVATE
    "${PROJECT_SOURCE_DIR}/."
)
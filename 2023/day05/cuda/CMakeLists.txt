cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(day5 LANGUAGES C CXX CUDA)

add_executable(
    ${PROJECT_NAME}
    main.cu
)
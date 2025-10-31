cmake_minimum_required(VERSION 3.25)

project(day5)

set(CMAKE_C_FLAGS "-O3 -fopenmp -g")

add_definitions("-DCL_TARGET_OPENCL_VERSION=300")

add_subdirectory("part_a/test")

add_test(NAME test_part_a COMMAND "part_a/test_part_a")

find_package(OpenMP)
find_package(OpenCL REQUIRED)

add_executable(
    "${PROJECT_NAME}"
    "${PROJECT_SOURCE_DIR}/main.c"
    "${PROJECT_SOURCE_DIR}/config/config.c"
    "${PROJECT_SOURCE_DIR}/part_a/part_a.c"
    "${PROJECT_SOURCE_DIR}/part_a/part_a_opencl.c"
    "${PROJECT_SOURCE_DIR}/part_a/part_a_openmp.c"
    "${PROJECT_SOURCE_DIR}/part_a/part_a_utils.c"
    "${PROJECT_SOURCE_DIR}/part_b/part_b.c"
    "${PROJECT_SOURCE_DIR}/part_b/part_b_openmp.c"
    "${PROJECT_SOURCE_DIR}/part_b/part_b_opencl.c"
    "${PROJECT_SOURCE_DIR}/part_b/part_b_utils.c"
    "${PROJECT_SOURCE_DIR}/seed_map/seed_map.c"
    "${PROJECT_SOURCE_DIR}/seed_map_layer/seed_map_layer.c"
    "${PROJECT_SOURCE_DIR}/seed_range/seed_range.c"
    "${PROJECT_SOURCE_DIR}/utils/utils.c"

)

target_include_directories(
    "${PROJECT_NAME}"
    PRIVATE
    "${PROJECT_SOURCE_DIR}/config/"
    "${PROJECT_SOURCE_DIR}/part_a/"
    "${PROJECT_SOURCE_DIR}/part_b/"
    "${PROJECT_SOURCE_DIR}/seed_map/"
    "${PROJECT_SOURCE_DIR}/seed_map_layer/"
    "${PROJECT_SOURCE_DIR}/seed_range/"
    "${PROJECT_SOURCE_DIR}/utils/"
)

target_link_libraries(
    "${PROJECT_NAME}" 
    PUBLIC 
    "OpenMP::OpenMP_CXX"
    "OpenCL::OpenCL"
)

enable_testing()
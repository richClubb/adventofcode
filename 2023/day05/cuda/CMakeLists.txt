cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(day5 LANGUAGES C CXX CUDA)

find_package(CUDA)

# can we add in a test target to run all the other tests?

add_executable(
    ${PROJECT_NAME}
    "${PROJECT_SOURCE_DIR}/main.cu"
    "${PROJECT_SOURCE_DIR}/config/config.cu"
    "${PROJECT_SOURCE_DIR}/part_a/part_a.cu"
    "${PROJECT_SOURCE_DIR}/part_b/part_b.cu"
    "${PROJECT_SOURCE_DIR}/seed/seed.cu"
    "${PROJECT_SOURCE_DIR}/seed_map/seed_map.cu"
    "${PROJECT_SOURCE_DIR}/seed_map_layer/seed_map_layer.cu"
    "${PROJECT_SOURCE_DIR}/seed_range/seed_range.cu"
    "${PROJECT_SOURCE_DIR}/utils/utils.cu"
)

target_include_directories(
    "${PROJECT_NAME}"
    PRIVATE
    "${PROJECT_SOURCE_DIR}/config/"
    "${PROJECT_SOURCE_DIR}/part_a/"
    "${PROJECT_SOURCE_DIR}/part_b/"
    "${PROJECT_SOURCE_DIR}/seed/"
    "${PROJECT_SOURCE_DIR}/seed_map/"
    "${PROJECT_SOURCE_DIR}/seed_map_layer/"
    "${PROJECT_SOURCE_DIR}/seed_range/"
    "${PROJECT_SOURCE_DIR}/utils/"
)
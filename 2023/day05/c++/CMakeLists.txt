cmake_minimum_required(VERSION 3.28)

project(day5)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp")

find_package(OpenMP)

add_executable(
    "${PROJECT_NAME}"
    "${PROJECT_SOURCE_DIR}/main.cpp"
    "${PROJECT_SOURCE_DIR}/config/config.cpp"
    "${PROJECT_SOURCE_DIR}/part_a/part_a.cpp"
    "${PROJECT_SOURCE_DIR}/part_b/part_b.cpp"
    "${PROJECT_SOURCE_DIR}/seed_map/seed_map.cpp"
    "${PROJECT_SOURCE_DIR}/seed_map_layer/seed_map_layer.cpp"
    "${PROJECT_SOURCE_DIR}/seed_range/seed_range.cpp"
    "${PROJECT_SOURCE_DIR}/utils/utils.cpp"

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
)
cmake_minimum_required(VERSION 3.25)

project(day5)

add_executable(
    "${PROJECT_NAME}"
    "${PROJECT_SOURCE_DIR}/main.c"
    "${PROJECT_SOURCE_DIR}/config/config.c"
    "${PROJECT_SOURCE_DIR}/part_a/part_a.c"
    "${PROJECT_SOURCE_DIR}/part_b/part_b.c"
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
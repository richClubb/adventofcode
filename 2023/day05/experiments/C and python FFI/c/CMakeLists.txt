cmake_minimum_required(VERSION 3.25)

project(day5 LANGUAGES C)

set(CMAKE_C_FLAGS                "-Wall")
set(CMAKE_C_FLAGS_DEBUG          "-pg -fopenmp")
set(CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE        "-O4 -march=native -funroll-loops -mtune=native -fno-exceptions -fopenmp -flto")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O4 -pg -fopenmp -flto")

set(LIB_PATH "/workspaces/adventofcode/2023/day05/experiments/C and python FFI/c_lib_src")

add_subdirectory(
    "${LIB_PATH}"
    "${LIB_PATH}/build/"
    EXCLUDE_FROM_ALL
)

add_executable(
    "${PROJECT_NAME}"
    "${PROJECT_SOURCE_DIR}/main.c"
    "${PROJECT_SOURCE_DIR}/config/config.c"
    "${PROJECT_SOURCE_DIR}/part_a/part_a.c"
    "${PROJECT_SOURCE_DIR}/part_b/part_b.c"
    "${PROJECT_SOURCE_DIR}/utils/utils.c"
)

target_include_directories(
    "${PROJECT_NAME}"
    PRIVATE
    "${PROJECT_SOURCE_DIR}/"
    "${PROJECT_SOURCE_DIR}/config/"
    "${PROJECT_SOURCE_DIR}/part_a/"
    "${PROJECT_SOURCE_DIR}/part_b/"
    "${PROJECT_SOURCE_DIR}/utils/"
)

target_link_directories(
    "${PROJECT_NAME}"
    PRIVATE
    "${LIB_PATH}/build/"
)

target_link_libraries(
    "${PROJECT_NAME}" 
    PUBLIC
    seed_calc
)

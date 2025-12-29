cmake_minimum_required(VERSION 3.25)

project(seed_calc LANGUAGES C)

set(CMAKE_C_FLAGS                "-Wall")
set(CMAKE_C_FLAGS_DEBUG          "-pg -flto")
set(CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE        "-O4 -march=native -funroll-loops -mtune=native -fno-exceptions -flto")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O4 -pg -flto -march=native -mtune=native")

add_library(
    ${PROJECT_NAME}
    SHARED
    "${PROJECT_SOURCE_DIR}/seed_calc.c"
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    "${PROJECT_SOURCE_DIR}/"
)
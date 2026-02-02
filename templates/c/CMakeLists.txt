cmake_minimum_required(VERSION 3.25)

project(day01 LANGUAGES C)

if(CLANG)
    # This currently doesn't work
    message("Clang Build")
    set(CMAKE_C_COMPILER             "/usr/bin/clang")
    set(CMAKE_C_FLAGS                "-Wall")
    set(CMAKE_C_FLAGS_DEBUG          "-g -fopenmp=libomp")
    set(CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG -fopenmp=libomp")
    set(CMAKE_C_FLAGS_RELEASE        "-O4 -DNDEBUG -fopenmp=libomp -flto")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O4 -pg -fopenmp=libomp -flto")

    set(CMAKE_AR      "/usr/bin/llvm-ar")
    set(CMAKE_LINKER  "/usr/bin/llvm-ld")
    set(CMAKE_NM      "/usr/bin/llvm-nm")
    set(CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
    set(CMAKE_RANLIB  "/usr/bin/llvm-ranlib")
else()
    set(CMAKE_C_FLAGS                "-Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG          "-g -fopenmp")
    set(CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE        "-O4 -march=native -funroll-loops -mtune=native -fno-exceptions -fopenmp -flto")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O4 -pg -fopenmp -flto")
endif()

add_definitions("-DCL_TARGET_OPENCL_VERSION=300")

find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(
    "${PROJECT_NAME}"
    "${PROJECT_SOURCE_DIR}/src/main.c"
    "${PROJECT_SOURCE_DIR}/src/config/config.c"
    "${PROJECT_SOURCE_DIR}/src/part_a/part_a.c"
    "${PROJECT_SOURCE_DIR}/src/part_b/part_b.c"
)

target_include_directories(
    "${PROJECT_NAME}"
    PRIVATE
    "${PROJECT_SOURCE_DIR}/src/"
    "${PROJECT_SOURCE_DIR}/src/config/"
    "${PROJECT_SOURCE_DIR}/src/part_a/"
    "${PROJECT_SOURCE_DIR}/src/part_b/"
)

target_link_libraries(
    "${PROJECT_NAME}" 
    PUBLIC 
    "OpenMP::OpenMP_C"
    "OpenCL::OpenCL"
)

cmake_minimum_required(VERSION 3.28)

project(day5)

if(CLANG)
    # Currently doesn't work :(
    set(CMAKE_CXX_COMPILER             "/usr/bin/clang++")
    set(CMAKE_CXX_FLAGS                "-Wall")
    set(CMAKE_CXX_FLAGS_DEBUG          "-g -fopenmp=libomp")
    set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG -fopenmp=libomp")
    set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG -fopenmp=libomp -flto")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -fopenmp=libomp -flto")

    set(CMAKE_AR      "/usr/bin/llvm-ar")
    set(CMAKE_LINKER  "/usr/bin/llvm-ld")
    set(CMAKE_NM      "/usr/bin/llvm-nm")
    set(CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
    set(CMAKE_RANLIB  "/usr/bin/llvm-ranlib")
else()
    message("GCC Build")
    set(CMAKE_CXX_FLAGS                "-Wall")
    set(CMAKE_CXX_FLAGS_DEBUG          "-pg -fopenmp")
    set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG -fopenmp")
    set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG -fopenmp -flto")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -flto")
endif()

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
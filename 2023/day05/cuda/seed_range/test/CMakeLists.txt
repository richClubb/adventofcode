cmake_minimum_required(VERSION 3.25)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")

project(test_seed_range LANGUAGES CUDA C CXX)

enable_testing()

add_executable(
  "${PROJECT_NAME}"
  test_seed_range.cu
  "${PROJECT_SOURCE_DIR}/../seed_range.cu"
  "${PROJECT_SOURCE_DIR}/../../utils/utils.cu"
)

target_include_directories(
  "${PROJECT_NAME}"
  PRIVATE
  "${PROJECT_SOURCE_DIR}/../"
  "${PROJECT_SOURCE_DIR}/../../utils/"
)

target_link_libraries(
  "${PROJECT_NAME}"
  cunit
)

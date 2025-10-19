cmake_minimum_required(VERSION 3.14)
project(test_part_a LANGUAGES C CXX CUDA)

enable_testing()

add_executable(
  "${PROJECT_NAME}"
  "test_part_a.cu"
  "${PROJECT_SOURCE_DIR}/../part_a.cu"
  "${PROJECT_SOURCE_DIR}/../../config/config.cu"
)

target_include_directories(
  "${PROJECT_NAME}"
  PRIVATE
  "${PROJECT_SOURCE_DIR}/../"
  "${PROJECT_SOURCE_DIR}/../../config"
)

target_link_libraries(
  "${PROJECT_NAME}"
  cunit
)

cmake_minimum_required(VERSION 3.14)
project(test_utils LANGUAGES C CXX CUDA)

enable_testing()

add_executable(
  "${PROJECT_NAME}"
  "test_utils.cu"
  "${PROJECT_SOURCE_DIR}/../utils.cu"
)

target_include_directories(
  "${PROJECT_NAME}"
  PRIVATE
  "${PROJECT_SOURCE_DIR}/../"
)

target_link_libraries(
  "${PROJECT_NAME}"
  cunit
)

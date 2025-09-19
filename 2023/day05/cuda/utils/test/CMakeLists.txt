cmake_minimum_required(VERSION 3.14)
project(test_utils)

enable_testing()

add_executable(
  "${PROJECT_NAME}"
  test_utils.c
  "${PROJECT_SOURCE_DIR}/../utils.c"
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

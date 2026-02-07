cmake_minimum_required(VERSION 3.14)
project(test_config)

enable_testing()

add_executable(
  "${PROJECT_NAME}"
  test_config.c
  "${PROJECT_SOURCE_DIR}/../config.c"
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

cmake_minimum_required(VERSION 3.14)
project(test_seed_map)

enable_testing()

add_executable(
  "${PROJECT_NAME}"
  test_seed_map.c
  "${PROJECT_SOURCE_DIR}/../seed_map.c"
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

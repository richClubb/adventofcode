cmake_minimum_required(VERSION 3.14)
project(test_part_b)

enable_testing()

add_executable(
  "${PROJECT_NAME}"
  test_part_b.c
  "${PROJECT_SOURCE_DIR}/../part_b.c"
  "${PROJECT_SOURCE_DIR}/../part_b_utils.c"
  "${PROJECT_SOURCE_DIR}/../../seed_map/seed_map.c"
  "${PROJECT_SOURCE_DIR}/../../seed_map_layer/seed_map_layer.c"
  "${PROJECT_SOURCE_DIR}/../../seed_range/seed_range.c"
  "${PROJECT_SOURCE_DIR}/../../utils/utils.c"
)

target_include_directories(
  "${PROJECT_NAME}"
  PRIVATE
  "${PROJECT_SOURCE_DIR}/../"
  "${PROJECT_SOURCE_DIR}/../../config/"
  "${PROJECT_SOURCE_DIR}/../../seed_map/"
  "${PROJECT_SOURCE_DIR}/../../seed_map_layer/"
  "${PROJECT_SOURCE_DIR}/../../seed_range/"
  "${PROJECT_SOURCE_DIR}/../../utils/"
)

target_link_libraries(
  "${PROJECT_NAME}"
  cunit
)

cmake_minimum_required(VERSION 3.14)
project(test_part_b LANGUAGES C CXX CUDA)

enable_testing()

add_executable(
  "${PROJECT_NAME}"
  "test_part_b.cu"
  "${PROJECT_SOURCE_DIR}/../part_b.cu"
  "${PROJECT_SOURCE_DIR}/../../config/config.cu"
  "${PROJECT_SOURCE_DIR}/../../seed/seed.cu"
  "${PROJECT_SOURCE_DIR}/../../seed_range/seed_range.cu"
  "${PROJECT_SOURCE_DIR}/../../seed_map/seed_map.cu"
  "${PROJECT_SOURCE_DIR}/../../seed_map_layer/seed_map_layer.cu"
  "${PROJECT_SOURCE_DIR}/../../utils/utils.cu"
)

target_include_directories(
  "${PROJECT_NAME}"
  PRIVATE
  "${PROJECT_SOURCE_DIR}/../"
  "${PROJECT_SOURCE_DIR}/../../config/"
  "${PROJECT_SOURCE_DIR}/../../seed/"
  "${PROJECT_SOURCE_DIR}/../../seed_range/"
  "${PROJECT_SOURCE_DIR}/../../seed_map/"
  "${PROJECT_SOURCE_DIR}/../../seed_map_layer/"
  "${PROJECT_SOURCE_DIR}/../../utils/"
)

target_link_libraries(
  "${PROJECT_NAME}"
  cunit
)

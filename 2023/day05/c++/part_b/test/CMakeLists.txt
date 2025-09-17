cmake_minimum_required(VERSION 3.14)
project(test_part_b)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  "${PROJECT_NAME}"
  "${PROJECT_SOURCE_DIR}/test_part_b.cpp"
  "${PROJECT_SOURCE_DIR}/../part_b.cpp"
  "${PROJECT_SOURCE_DIR}/../../utils/utils.cpp"
  "${PROJECT_SOURCE_DIR}/../../seed_range/seed_range.cpp"
  "${PROJECT_SOURCE_DIR}/../../seed_map_layer/seed_map_layer.cpp"
  "${PROJECT_SOURCE_DIR}/../../seed_map/seed_map.cpp"
)

target_include_directories(
  "${PROJECT_NAME}"
  PRIVATE
  "${PROJECT_SOURCE_DIR}/.."
  "${PROJECT_SOURCE_DIR}/../../config"
  "${PROJECT_SOURCE_DIR}/../../utils"
  "${PROJECT_SOURCE_DIR}/../../seed_map"
  "${PROJECT_SOURCE_DIR}/../../seed_range"
  "${PROJECT_SOURCE_DIR}/../../seed_map_layer"
)

target_link_libraries(
  "${PROJECT_NAME}"
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests("${PROJECT_NAME}")
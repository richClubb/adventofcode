#include <gtest/gtest.h>

#include "config.h"

#include "part_b.h"

// Demonstrate some basic assertions.
TEST(PartATest, PtrVersionTest) {

  CONFIG config = {
    .input_file_path = "/workspaces/adventofcode/2023/day05/c++/part_b/test/part_b_sample.txt"
  };

  uint64_t result = part_b_ptr_version(config);

  EXPECT_EQ(result, 46);
}

// Demonstrate some basic assertions.
TEST(PartATest, OptVersionTest) {

  CONFIG config = {
    .input_file_path = "/workspaces/adventofcode/2023/day05/c++/part_b/test/part_b_sample.txt"
  };

  uint64_t result = part_b_optional_version(config);

  EXPECT_EQ(result, 46);
}
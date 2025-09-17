#include <gtest/gtest.h>

#include "part_a.h"

#include <stdint.h>

// Demonstrate some basic assertions.
TEST(PartATest, SampleRun1) {

  CONFIG config = {
    .input_file_path = "/workspaces/adventofcode/2023/day05/c++/part_a/test/part_a_sample.txt"
  };

  uint64_t result = part_a(config);

  EXPECT_EQ(result, 35);
}
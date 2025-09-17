#include <gtest/gtest.h>

#include "config.h"

#include "part_b.h"

// Demonstrate some basic assertions.
TEST(PartATest, SampleRun1) {

  CONFIG config = {
    .input_file_path = "/workspaces/adventofcode/2023/day05/c++/part_b/test/part_b_sample.txt"
  };

  uint64_t result = part_b(config);

  EXPECT_EQ(result, 46);
}
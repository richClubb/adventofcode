#include <gtest/gtest.h>

#include "part_a.h"

#include <stdint.h>

// Demonstrate some basic assertions.
TEST(PartATest, SampleRun1) {

  CONFIG config = {
    .input_file_path = "../part_a_sample.txt"
  };

  uint32_t result = part_a(config);

}
#include <gtest/gtest.h>

#include <stdint.h>

#include <vector>

#include "utils.h"

// Demonstrate some basic assertions.
TEST(UtilsTests, ExtractNumbers) {
    
    std::vector<uint32_t> numbers = extract_numbers("1 2 5 10 100\n");

    EXPECT_EQ(numbers.size(), 5);

    EXPECT_EQ(numbers[0], 1);
    EXPECT_EQ(numbers[1], 2);
    EXPECT_EQ(numbers[2], 5);
    EXPECT_EQ(numbers[3], 10);
    EXPECT_EQ(numbers[4], 100);
}
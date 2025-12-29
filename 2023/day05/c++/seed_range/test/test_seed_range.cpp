#include <gtest/gtest.h>

#include <stdint.h>

#include "seed_range.h"

TEST(SeedMapLayerTests, GetSeedRanges) {
  
  std::vector<SeedRange> seed_ranges = get_seed_ranges("seeds: 1 2 3 4 5 6\n");

  EXPECT_EQ(seed_ranges.size(), 3);

  EXPECT_EQ(seed_ranges[0].get_start(), 1);
  EXPECT_EQ(seed_ranges[0].get_size(),  2);
  EXPECT_EQ(seed_ranges[0].get_end(),   2);

  EXPECT_EQ(seed_ranges[1].get_start(), 3);
  EXPECT_EQ(seed_ranges[1].get_size(),  4);
  EXPECT_EQ(seed_ranges[1].get_end(),   6);

  EXPECT_EQ(seed_ranges[2].get_start(), 5);
  EXPECT_EQ(seed_ranges[2].get_size(),  6);
  EXPECT_EQ(seed_ranges[2].get_end(),   10);
}
#include <gtest/gtest.h>

#include <stdint.h>

#include "seed_map.h"

// Demonstrate some basic assertions.
TEST(SeedMapTests, BasicConstructors) {
  // Expect two strings not to be equal.
  SeedMap seed_map(1, 10, 5);

  SeedMap *seed_map_ptr_1 = new SeedMap(10, 20, 6);

  SeedMap *seed_map_ptr_2 = new SeedMap("1 5 5\n");
  SeedMap *seed_map_ptr_3 = new SeedMap("10 15 2\0");

  delete seed_map_ptr_1;
  delete seed_map_ptr_2;
  delete seed_map_ptr_3;
}

TEST(SeedMapTests, MapSeed)
{
  SeedMap seed_map(2, 10, 5);

  {
    uint64_t input = 1;
    EXPECT_EQ(seed_map.map_seed(&input), false);
    EXPECT_EQ(input, 1);
  }
  {
    uint64_t input = 2;
    EXPECT_EQ(seed_map.map_seed(&input), true);
    EXPECT_EQ(input, 10);
  }
  {
    uint64_t input = 3;
    EXPECT_EQ(seed_map.map_seed(&input), true);
    EXPECT_EQ(input, 11);
  }
  {
    uint64_t input = 4;
    EXPECT_EQ(seed_map.map_seed(&input), true);
    EXPECT_EQ(input, 12);
  }
  {
    uint64_t input = 5;
    EXPECT_EQ(seed_map.map_seed(&input), true);
    EXPECT_EQ(input, 13);
  }
  {
    uint64_t input = 6;
    EXPECT_EQ(seed_map.map_seed(&input), true);
    EXPECT_EQ(input, 14);
  }
  {
    uint64_t input = 7;
    EXPECT_EQ(seed_map.map_seed(&input), false);
    EXPECT_EQ(input, 7);
  }
}
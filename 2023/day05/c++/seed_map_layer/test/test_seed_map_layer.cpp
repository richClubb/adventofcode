#include <gtest/gtest.h>

#include <stdint.h>

#include "seed_map.h"
#include "seed_map_layer.h"

// Demonstrate some basic assertions.
TEST(SeedMapLayerTests, BasicConstructors) {
  // Expect two strings not to be equal.
  SeedMapLayer seed_map_layer_1;
  SeedMapLayer seed_map_layer_2("test");

  SeedMapLayer *seed_map_layer_ptr_1 = new SeedMapLayer();
  SeedMapLayer *seed_map_layer_ptr_2 = new SeedMapLayer("test");

  delete seed_map_layer_ptr_1;
  delete seed_map_layer_ptr_2;
}

TEST(SeedMapLayerTests, AddMaps)
{
  SeedMapLayer *seed_map_layer = new SeedMapLayer();

  SeedMap seed_map_1(1, 10, 5);

  seed_map_layer->add_seed_map(seed_map_1);
  seed_map_layer->add_seed_map(SeedMap(20, 15, 2));

  delete seed_map_layer;
}

TEST(SeedMapLayerTests, MapSeed)
{
  SeedMapLayer *seed_map_layer = new SeedMapLayer();

  seed_map_layer->add_seed_map(SeedMap(2, 10, 5));

  std::optional<uint64_t> result = seed_map_layer->map_seed(1);
  EXPECT_EQ(result, std::nullopt);

  result = seed_map_layer->map_seed(2);
  EXPECT_EQ(result, 10);

  result = seed_map_layer->map_seed(3);
  EXPECT_EQ(result, 11);

  result = seed_map_layer->map_seed(4);
  EXPECT_EQ(result, 12);

  result = seed_map_layer->map_seed(5);
  EXPECT_EQ(result, 13);

  result = seed_map_layer->map_seed(6);
  EXPECT_EQ(result, 14);

  result = seed_map_layer->map_seed(7);
  EXPECT_EQ(result, std::nullopt);

  delete seed_map_layer;
}

TEST(SeedMapLayerTests, SortSeedMaps)
{
  SeedMapLayer seed_map_layer;

  seed_map_layer.add_seed_map(SeedMap(1, 2, 5));
  seed_map_layer.add_seed_map(SeedMap(5, 7, 5));
  seed_map_layer.add_seed_map(SeedMap(4, 1, 5));

  seed_map_layer.sort_seed_maps();

  return;

}
#ifndef __SEED_MAP_LAYER_H__

#define __SEED_MAP_LAYER_H__

#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

#include "seed_map.h"

class SeedMapLayer
{
public:
    SeedMapLayer();
    SeedMapLayer(std::string);
    ~SeedMapLayer();

    void add_seed_map(SeedMap);

    std::optional<uint64_t> map_seed(uint64_t input);

    void sort_seed_maps();

private:
    std::vector<SeedMap> seed_maps;
    std::string name;

};

#endif
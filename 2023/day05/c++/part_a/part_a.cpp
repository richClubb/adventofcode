#include "part_a.h"

#include <stdint.h>

#include "config.h"
#include <limits.h>

#include <iostream>
#include <fstream>
#include <vector>

#include "utils.h"
#include "seed_map_layer.h"

uint64_t part_a(const CONFIG &config)
{
    std::ifstream input_file(config.input_file_path);

    std::vector<uint64_t> seeds;
    std::vector<SeedMapLayer> seed_map_layers;
    SeedMapLayer *curr_layer;

    std::string line;

    while(getline(input_file, line))
    {
        if( line.length() < 2 )
        {
            continue;
        }

        if( line.find("seeds: ") != std::string::npos)
        {
            seeds = get_seeds(line);
            continue;
        }

        if (line.find(":") != std::string::npos)
        {
            seed_map_layers.push_back(SeedMapLayer(line));
            curr_layer = &seed_map_layers.back();
            continue;
        }

        curr_layer->add_seed_map(SeedMap(line));
    }

    uint64_t min_value = UINT64_MAX;
    for(const auto &seed : seeds)
    {
        uint64_t value = seed;
        for (uint64_t index = 0; index < seed_map_layers.size(); index++)
        {
            seed_map_layers[index].map_seed(&value);
        }

        if (value < min_value)
        {
            min_value = value;
        }
    }
    
    return min_value;
}
#include "part_b.h"

#include <stdint.h>

#include "config.h"
#include <limits.h>

#include <iostream>
#include <fstream>
#include <vector>

#include "utils.h"
#include "seed_range.h"
#include "seed_map_layer.h"

uint64_t part_b(const CONFIG &config)
{
    std::ifstream input_file(config.input_file_path);

    std::vector<SeedRange> seed_ranges;
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
            seed_ranges = get_seed_ranges(line);
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
    
    for(SeedRange &seed_range : seed_ranges)
    {
        uint64_t start = seed_range.get_start();
        uint64_t size = seed_range.get_size();
        uint64_t end = seed_range.get_end();
        uint64_t range_min = UINT64_MAX;

        for(uint64_t seed = start; seed <= end; seed++)
        {
            uint64_t value = seed;
            for (uint64_t index = 0; index < seed_map_layers.size(); index++)
            {
                if (std::optional<uint64_t> result; result = seed_map_layers[index].map_seed(value))
                {
                    value = result.value();
                }
            }

            if (value < range_min)
            {
                range_min = value;
            }    
        }
        if (range_min < min_value)
        {
            min_value = range_min;
        }
    }
    
    return min_value;
}
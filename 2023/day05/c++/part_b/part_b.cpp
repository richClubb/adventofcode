#include "part_b.h"

#include <stdint.h>
#include <omp.h>

#include "config.h"
#include <limits.h>

#include <iostream>
#include <fstream>
#include <vector>

#include "utils.h"
#include "seed_range.h"
#include "seed_map_layer.h"

std::tuple<std::vector<SeedRange>, std::vector<SeedMapLayer>> injest_file_part_b(const char *input_file_path)
{
    std::ifstream input_file(input_file_path);

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

    return std::make_tuple(seed_ranges, seed_map_layers);
}

uint64_t part_b_ptr_version(const CONFIG &config)
{
    std::vector<SeedRange> seed_ranges;
    std::vector<SeedMapLayer> seed_map_layers;

    tie(seed_ranges, seed_map_layers) = injest_file_part_b(config.input_file_path);

    std::vector<uint64_t> results(seed_ranges.size(), 0);
    
    for(uint64_t seed_range_index = 0; seed_range_index < seed_ranges.size(); seed_range_index++)
    {
        SeedRange *seed_range = &seed_ranges[seed_range_index];
        uint64_t start = seed_range->get_start();
        uint64_t size = seed_range->get_size();
        uint64_t end = seed_range->get_end();
        uint64_t range_min = UINT64_MAX;

        for(uint64_t seed = start; seed <= end; seed++)
        {
            uint64_t value = seed;
            for (uint64_t index = 0; index < seed_map_layers.size(); index++)
            {
                seed_map_layers[index].map_seed(&value);
            }

            if (value < range_min) range_min = value;
        }

        results[seed_range_index] = range_min;
    }
    
    uint64_t min_value = UINT64_MAX;

    for(uint64_t result_index = 0; result_index < results.size(); result_index++)
    {
        if (results[result_index] < min_value) min_value = results[result_index];
    }

    return min_value;
}

uint64_t part_b_openmp(const CONFIG &config)
{
    std::vector<SeedRange> seed_ranges;
    std::vector<SeedMapLayer> seed_map_layers;

    tie(seed_ranges, seed_map_layers) = injest_file_part_b(config.input_file_path);

    std::vector<uint64_t> results(seed_ranges.size(), 0);
    
    #pragma omp parallel for num_threads(28)
    for(uint64_t seed_range_index = 0; seed_range_index < seed_ranges.size(); seed_range_index++)
    {
        SeedRange *seed_range = &seed_ranges[seed_range_index];
        uint64_t start = seed_range->get_start();
        uint64_t size = seed_range->get_size();
        uint64_t end = seed_range->get_end();
        uint64_t range_min = UINT64_MAX;

        for(uint64_t seed = start; seed <= end; seed++)
        {
            uint64_t value = seed;
            for (uint64_t index = 0; index < seed_map_layers.size(); index++)
            {
                seed_map_layers[index].map_seed(&value);
            }

            if (value < range_min) range_min = value;
        }

        results[seed_range_index] = range_min;
    }
    
    uint64_t min_value = UINT64_MAX;

    for(uint64_t result_index = 0; result_index < results.size(); result_index++)
    {
        if (results[result_index] < min_value) min_value = results[result_index];
    }

    return min_value;
}

uint64_t part_b_optional_version(const CONFIG &config)
{
    std::vector<SeedRange> seed_ranges;
    std::vector<SeedMapLayer> seed_map_layers;

    tie(seed_ranges, seed_map_layers) = injest_file_part_b(config.input_file_path);

    uint64_t min_value = UINT64_MAX;
    
    for(SeedRange &seed_range : seed_ranges)
    {
        uint64_t start = seed_range.get_start();
        uint64_t size = seed_range.get_size();
        uint64_t end = seed_range.get_end();

        for(uint64_t seed = start; seed <= end; seed++)
        {
            std::optional<uint64_t> value = seed;
            for (uint64_t index = 0; index < seed_map_layers.size(); index++)
            {
                value = seed_map_layers[index].map_seed_opt(value.value());
            }

            if (value < min_value) min_value = value.value();   
        }
    }
    
    return min_value;
}
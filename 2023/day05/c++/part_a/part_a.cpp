#include "part_a.h"

#include <stdint.h>

#include "config.h"
#include <limits.h>
#include <omp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#include "utils.h"
#include "seed_map_layer.h"

std::tuple<std::vector<uint64_t>, std::vector<SeedMapLayer>> injest_file_part_a(const char *input_file_path)
{
    std::ifstream input_file(input_file_path);

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

    return std::make_tuple(seeds, seed_map_layers);
}

uint64_t part_a(const CONFIG &config)
{
    std::vector<uint64_t> seeds;
    std::vector<SeedMapLayer> seed_map_layers;

    tie(seeds, seed_map_layers) = injest_file_part_a(config.input_file_path);

    std::vector<uint64_t> results(0, seeds.size());
    for (uint64_t seed_index = 0; seed_index < seeds.size(); seed_index++)
    {
        uint64_t value = seeds[seed_index];
        for (uint64_t index = 0; index < seed_map_layers.size(); index++)
        {
            seed_map_layers[index].map_seed(&value);
        }

        results[seed_index] = value;
    }
    
    uint64_t min_value = UINT64_MAX;
    for (uint64_t result_index = 0; result_index < seeds.size(); result_index++)
    {
        if (results[result_index] < min_value) min_value = results[result_index];
    }

    return min_value;
}

uint64_t part_a_openmp(const CONFIG &config)
{
    std::vector<uint64_t> seeds;
    std::vector<SeedMapLayer> seed_map_layers;

    tie(seeds, seed_map_layers) = injest_file_part_a(config.input_file_path);

    std::vector<uint64_t> results(seeds.size(), 0);

    #pragma omp parallel for num_threads(28)
    for (uint64_t seed_index = 0; seed_index < seeds.size(); seed_index++)
    {
        uint64_t value = seeds[seed_index];
        for (uint64_t index = 0; index < seed_map_layers.size(); index++)
        {
            seed_map_layers[index].map_seed(&value);
        }

        results[seed_index] = value;
    }
    
    uint64_t min_value = UINT64_MAX;
    for (uint64_t result_index = 0; result_index < seeds.size(); result_index++)
    {
        if (results[result_index] < min_value) min_value = results[result_index];
    }

    return min_value;
}


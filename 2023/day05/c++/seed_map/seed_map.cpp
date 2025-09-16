#include "seed_map.h"

#include <assert.h>

#include <bits/stdc++.h>
#include <optional>

SeedMap::SeedMap(uint32_t source, uint32_t target, uint32_t size)
{
    this->source = source;
    this->target = target;
    this->size = size;
}

SeedMap::SeedMap(std::string input_string)
{
    std::stringstream ss(input_string);

    std::vector<uint32_t> numbers;

    std::string token;
    while(getline(ss, token, ' '))
    {
        numbers.push_back(std::stol(token, 0, 10));
    }

    assert(numbers.size() == 3);

    this->source = numbers[1];
    this->target = numbers[0];
    this->size = numbers[2];
}

SeedMap::~SeedMap()
{
    this->source = 0;
    this->target = 0;
    this->size = 0;
}

std::optional<uint32_t> SeedMap::map_seed(uint32_t input)
{
    if (
        (input >= this->source) && (input < this->source + this->size)
    )
    {
        return (input - this->source) + this->target;
    }

    return std::nullopt;
}
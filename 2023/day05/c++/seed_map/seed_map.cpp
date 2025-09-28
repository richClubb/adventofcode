#include "seed_map.h"

#include <stdio.h>
#include <assert.h>

#include <bits/stdc++.h>
#include <optional>

SeedMap::SeedMap(uint64_t source, uint64_t target, uint64_t size)
{
    this->source = source;
    this->source_end = source + size - 1;
    this->target = target;
    this->target_end = target + size - 1;
    this->size = size;
}

SeedMap::SeedMap(std::string input_string)
{
    std::stringstream ss(input_string);

    std::vector<uint64_t> numbers;

    std::string token;
    while(getline(ss, token, ' '))
    {
        numbers.push_back(std::stol(token, 0, 10));
    }

    assert(numbers.size() == 3);

    this->source = numbers[1];
    this->target = numbers[0];
    this->size = numbers[2];

    this->source_end = this->source + size - 1;
    this->target_end = this->target + size - 1;
}

SeedMap::~SeedMap()
{
    this->source = 0;
    this->target = 0;
    this->size = 0;
}

bool SeedMap::map_seed(uint64_t *input)
{
    if 
    (
        (*input >= this->source) && 
        (*input < this->source + this->size)
    )
    {
        *input = (*input - this->source) + this->target;
        return true;
    }

    return false;
}

std::optional<uint64_t> SeedMap::map_seed_opt(uint64_t input)
{
    if (
        (input >= this->source) &&
        (input < this->source + this->size)
    )
    {
        return input - this->source + this->target;
    }
    return std::nullopt;
}
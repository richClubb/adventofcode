#include "seed_range.h"

#include "utils.h"

SeedRange::SeedRange(uint64_t start, uint64_t size)
{
    this->start = start;
    this->size = size;
}

SeedRange::~SeedRange()
{
    this->start = 0;
    this->size = 0;
}

std::vector<SeedRange> get_seed_ranges(std::string input_string)
{
    std::vector<uint64_t> numbers = get_seeds(input_string);

    std::vector<SeedRange> seed_ranges;

    for( uint64_t index = 0; index < numbers.size(); index+=2)
    {
        seed_ranges.push_back(SeedRange(numbers[index], numbers[index+1]));
    }

    return seed_ranges;
}
#ifndef __SEED_RANGE_H__

#define __SEED_RANGE_H__

#include <stdint.h>

#include <vector>
#include <string>

#include "seed_range.h"

class SeedRange
{
public:
    SeedRange(uint64_t, uint64_t);
    ~SeedRange();

    uint64_t get_next_seed();
    void reset();

    uint64_t get_start(){ return this->start; }
    uint64_t get_size(){ return this->size; }
    uint64_t get_end(){ return this->start + this->size - 1; }

private:
    uint64_t start;
    uint64_t size;
};

std::vector<SeedRange> get_seed_ranges(std::string);

#endif
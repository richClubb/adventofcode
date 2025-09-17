#ifndef __SEED_MAP_H__

#define __SEED_MAP_H__

#include <stdint.h>

#include <optional>
#include <string>


class SeedMap
{
public:
    SeedMap(uint64_t, uint64_t, uint64_t);
    SeedMap(std::string);
    ~SeedMap();
    std::optional<uint64_t> map_seed(uint64_t);

    uint64_t get_source(){ return source; }

private:
    uint64_t source;
    uint64_t target;
    uint64_t size;
};

#endif
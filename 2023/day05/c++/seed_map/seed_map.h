#ifndef __SEED_MAP_H__

#define __SEED_MAP_H__

#include <stdint.h>

#include <optional>
#include <string>


class SeedMap
{
public:
    SeedMap(uint32_t, uint32_t, uint32_t);
    SeedMap(std::string);
    ~SeedMap();
    std::optional<uint32_t> map_seed(uint32_t);

private:
    uint32_t source;
    uint32_t target;
    uint32_t size;
};

#endif
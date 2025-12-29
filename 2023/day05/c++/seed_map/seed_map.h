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
    bool map_seed(uint64_t *);
    std::optional<uint64_t> map_seed_opt(uint64_t input);

    uint64_t get_source(){ return source; }
    uint64_t get_source_end(){ return this->source_end; }
    uint64_t get_target(){ return this->target; };
    uint64_t get_target_end(){ return this->target_end; };

private:
    uint64_t source;
    uint64_t source_end;
    uint64_t target;
    uint64_t target_end;
    uint64_t size;
};

#endif
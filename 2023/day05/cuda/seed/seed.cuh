#ifndef __SEED_H__

#define __SEED_H__

#include <stdint.h>

typedef uint64_t SEED;

SEED *seed_get_seeds_from_line(const char* line, uint64_t *num_seeds);

#endif
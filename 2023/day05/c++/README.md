Advent of code 2023 Day 05 - C++
--------------------------------

# Overview

This was interesting. Instead of just programming C in C++ i made sure that I used as many of the C++ features I could. This included using the `std::optional` types for the mapping functions. I really like the paradigm of using the `Result` type in Rust and this is pretty analogous.

The use of `std::vector` is very helpful in comparison to C and manual indexing and it's frustrating how much of a difference this makes to general programming.

## Performance impact of optionals

std::optional had a large performance penalty with unoptimised builds, but these went away when using a "release" build. You can see the difference in the [results](./../docs/results.xlsx). 

I wrote a "ptr" version which is the same as the C implementation and a "optional" version which I've also done in Rust using the `Result` type. The difference between the "ptr" (00:23:13) and "optional" (2:29:25) versions in the debug builds is pretty staggering, but they both were around the 1 - 2 minute mark when optimised. There does still seem to be a penalty, but I haven't dug into this in great detail.

I'd suggest looking at (https://godbolt.org/) and putting in the following snippet

```C++
#include <stdio.h>
#include <optional>

typedef struct seed_map_t {
    unsigned int source;
    unsigned int target;
    unsigned int size;
} SEED_MAP;

std::optional<unsigned int> test_optional(unsigned int number, SEED_MAP *seed_map)
{
    if (
        (number >= seed_map->source) && 
        (number < seed_map->source + seed_map->source + seed_map->size)
    )
    {
        return number - seed_map->source + seed_map->size;
    }

    return std::nullopt;
}

bool test_ptr(unsigned int *number, SEED_MAP *seed_map)
{
    if (
        (*number >= seed_map->source) && 
        (*number < seed_map->source + seed_map->source + seed_map->size)
    )
    {
        *number = *number - seed_map->source + seed_map->size;
        return true;
    }

    return false;
}
```

If you go through the "-O0" to "-O3" options you'll see the difference. This can also be seen with `gprof`. If you compile the solution with "-pg" and then assess the output with `gprof` you'll see the time taken for each call.

# Build / Run

```
mkdir build-x86
cd build-x86/
cmake ../
make
./day5 -i [path] -r [run]
```

Where [run] is:
* part_a
* part_a_openmp
* part_b_ptr
* part_b_optional
* part_b_openmp

To build for debug or release use the `-DCMAKE_BUILD_TYPE=[type]` parameter, where [type] is `Debug` or `Release`

# Test

Each module has its own test directory which contains its own unit test.

```
cd [module]/test/
mkdir build-x86
cd build-x86/
cmake ../
make
./[module]
```

replace [module] with; part_a, part_b, seed_map, seed_map_layer, seed_range, utils


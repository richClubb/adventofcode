# Advent of code 2023 Day 05 - C++

## To-do

* Write-up the performance impacts

## Build / Run

```
mkdir build-x86
cd build-x86/
cmake ../
make
./day5 -i [path]
```

## Test

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

std::optional had a 50% performance penalty

```cpp
for(SeedMap &seed_map : this->seed_maps)
{
    if (bool result; result = seed_map.map_seed(input))
    {
        return true;
    }
}
```

has a 10 second penalty over

```cpp
    for(uint64_t index = 0; index < this->seed_maps.size(); index++)
    {    
        if (bool result; result = this->seed_maps[index].map_seed(input))
        {
            return false;
        }
    }
```
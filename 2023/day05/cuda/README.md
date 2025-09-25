# Advent of code 2023 Day 05 - Cuda

## To-do

* figure out how many threads / blocks I can use
* run more in parallel

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
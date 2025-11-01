# Advent of code 2023 Day 05 - C

## To-do

## Build / Run

```
mkdir build-x86
cd build-x86/
cmake ../
make
./day5 -i [path] -r [run]
```

run can be: 
* `part_a`
* `part_a_openmp`
* `part_a_opencl`
* `part_b`
* `part_b_openmp`
* `part_b_opencl` - This one has some issues

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

### Questions asked

https://stackoverflow.com/questions/79806444/opencl-kernel-slow-and-doesnt-utilise-cpu-fully
Advent of code 2023 Day 05 - Cuda
---------------------------------

# Overview

Getting my head around this paradigm was fun, it's not too bad but it was good to have a problem that I had to figure out the solution for rather than just the examples.

Generally there is so much more setup than for any other version of this problem (other than OpenCL) but it's fairly boilerplate. The main consideration is that if you know you're going to use CUDA or OpenCL you have to format the datastructures in a way that makes sense. If you don't you're immediately in trouble as you have to wrangle the data into a format that can be passed to the GPU.

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
* part_a_non_kernel
* part_b
* part_b_non_kernel

I haven't done any optimisation of the cuda code, it seems to run the same with `-DCMAKE_BUILD_TYPE=Release` as `-DCMAKE_BUILD_TYPE=Debug`

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
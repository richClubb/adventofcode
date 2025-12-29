Advent of code 2023 Day 05 - C
------------------------------

# Overview

I broke this down into 3 separate solutions: C single threaded, C with OpenMP and C with OpenCL. The single threaded C was pretty easy for the actual solution but frustrating for parsing the code as it requires a lot of string wrangling.

I'm not great at memory management so I tried to be as dynamic as possible and using `valgrind` and `-fsanitize=address` to detect memory leaks and issues and they did pick up some issues which I fixed pretty easily.

Overall it was fun to do in C, would highly recommend, especially if you work more in higher level languages.

The package management with CMake was "ok". I'm not a huge fan of CMake but it's the best of a bad bunch from what I understand. CMake combined with vscode devcontainers is a pretty winning combination and it's easy enough to deploy a docker version, it can be an issue if you've got multiple program going onto the same platform but this can be managed with human processes.

## OpenCL and OpenMP

I wanted to try parallelising the code as I had with other languages and found there were 2 common methods for this, OpenCL and OpenMP. OpenCL is used primarily for using TPU, GPU or other accelerators but does work for CPU scaling and it's very similar paradigm to CUDA. OpenMP works using directives added to the code which can help to parallelise loops and other constructs so it's much easier to add into existing code without having to re-architect it, I'm not an expert at either so I'd recommend further reading. It does seem to have a much more significant overhead though, the difference in speed between the OpenCL (at 9 seconds) and OpenMP ( at 24 seconds ) was pretty significant.

The experience was fun overall, finding out that OpenMP can be used very easily was great, but the hassle of OpenCL was very frustrating, it's a great thing to know, and I imaging if it's possible to leverage the GPU as well then it can be very, very scalable, but for small problems it's not really worth it.

# Build / Run

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
* `part_b_opencl`

To build for debug or release use the `-DCMAKE_BUILD_TYPE=[type]` parameter, where [type] is `Debug` or `Release`

# Test

The testing for this code was done in CUnit, each module has its own test directory which contains its own unit test. They don't have 100% coverage but I did try to be disciplined about creating the tests alongside the code.

```
cd [module]/test/
mkdir build-x86
cd build-x86/
cmake ../
make
./[module]
```

replace [module] with; part_a, part_b, seed_map, seed_map_layer, seed_range, utils

# Future work

* OpenCL GPU on Nvidia, ATI and Intel GPUs

# Questions asked

https://stackoverflow.com/questions/79806444/opencl-kernel-slow-and-doesnt-utilise-cpu-fully - Answered this one with a little prodding.
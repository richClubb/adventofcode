# Languages

I decided to write this is a variety of languages to challenge myself to solve a problem in different languages idiomatically to practice. For this I chose:
* C
* C++
* Cuda
* F#
* Go
* Python
* Rust
* Zig

I'll go through each language and let you know what I thought of each

## Languge agnostic paradigms

### Devcontainers

I tried to set up devcontainers for the project so that it's easier to manage dependencies.



### Project organisation

I've tried to develop these examples with a heavy TDD approach with high levels of unit test coverage.

I've organised the projects into foldes that contain each module.

```
src/
  main.c
  CMakeLists.txt
  part_a/
    test/
      CMakeLists.txt
      part_a_sample.txt
      test_part_a.c
    part_a.c
    part_a.h
  seed_map/
    test/
      CMakeLists.txt
      test_seed_map.c
    seed_map.c
    seed_map.h
```

Rather than

```
include/
  seed_map.h
  part_a.h
src/
  main.c
  part_a.c
  seed_map.c
test/
  CMakeLists.txt
  test_part_a.c
  test_seed_map.c
```

Even though this requires more boilerplate code for the CMakeLists.txt I prefer the segmentation of the project. It separates each of the components of the code into separate components and each folder is a more self-contained system. I've tried to mirorr that in each one of the repos.

### Use of inline debuggers

I have tried to set up some debug targets in the `.vscode` folder for each project so that they can be reused inside the devcontainers

## C

I like C as a language, but OMG does it need some quality of life improvements. I tried to use just raw C with as few bells and whistles as possible but I didn't arbitrarily limit myself to just C99. If the C standards have been released then they are good enough for me.

### Toolchain

I've had to learn more about CMake for my current job as most of the build systems are done with CMake. It's good enough but it is very frustrating.

### Unit testing

I used CUnit for the unit testing framwork. It's pretty pleasant and easy to use.

## C++

I'm having to do more C++ for my job at the moment, and I'm finding it pretty pleasant. 

### GoogleTest

I used googletest for the unit testing and I'm not sure which one I prefer. I think that googletest has better support for mocking than Cunit.

### HORRIFIC PERFORMANCE IMPLICATIONS

I decided to use the `optional` keyword to return `nullptr` or a `uint64_t` value from the map seeds functions. The problem is that this cost a huge amount, almost doubling the execution time of the code. 

The use of vector iterators was also another serious performance hit which was hard to stomach in some cases.

## Cuda

This was my first foray into Cuda programming and it was a bit more pleasant than I'd anticipated although sharing memory between the CPU space and GPU space is something to consider carefully.

My solution 'works' but isn't very good and could definitely use some more work to make it good. I need to research how to leverage the blocks and threads more effectively to parallelise the computation and calculate the minimum on the fly better.

### Unit testing

I

## f#

This was an experiment to program specifically in a functional lanugage without using any imperative constructs, however it didn't work. For the sample it worked fine but when calculating the full values I ran into the bredth first search problem as it had to keep track of both the input vector and output vecfor which would have been 32Gb of memory

## go

Not done this one yet as I couldn't get the devcontainer feature to work correctly. May come back to this at a later date.

## Python

This was the first implementation which was quick and dirty. Worked well and was very useful for helping when I wanted to figure out the range based calculation. I really didn't want to have to do that in C

## Rust

A lot smoother than I'd anticipated. The use of the rayon crate and the 'par_iter' function was beautiful.

## Zig 

I don't like zig very much. I dislike the build system. The syntax isn't bad and I like the use of the test code inside the module.
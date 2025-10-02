# Languages

I decided to write this is a variety of languages to challenge myself to solve a problem in different languages idiomatically to practice. For this I chose:

* [C](./c/) - COMPLETE
* [C++](./c++/) - COMPLETE
* [Cuda]() - COMPLETE (ISH)
* [F#](./f_sharp/) - COMPLETE
* [Go](./..) - 
* [Python](./python/) - COMPLETE
* [Rust](./rust/) - COMPLETE
* [Zig](./zig/) - INCOMPLETE

I'll go through each language and let you know what I thought of each

## Performance Review

* C (not optimised) - 0m3.997s
* C (optimised `-O3`) - 0m1.567s
* C++ ptr version (not optimised) - 0m6.210s
* C++ ptr version (optimised `-O3`) - 0m2.010s
* C++ optional version (not optimised) - 0m24.228s
* C++ optional version (optimised `-O3`) - 0m2.011s
* F# (not optimised) - 0m17.259s
* Go (run through `go run`) - 0m1.936s
* Python (3.12) - 5m18.282s
* Rust (not optimised) - 0m8.356s
* Rust (release version) - 0m0.622s ( WINNER !!!!!!! )
* Zig (not optimised) - 1m46.424s


## Enjoyment Rating

* Rust - was really nice to learn a more functional way of doing things. The parallelisation tooling is pretty wicked.
* Cuda - Frustrating. You really have to think about the problems differently, and memory management between the CPU and GPU is a PITA, but it was interesting to learn.
* C - Felt familiar, and was fun to use a lot of dynamic memory allocation, although the unit testing / management was a PITA
* C++ - Interesting investigation into the use of the optional type, and interesting to see the difference when performance optimisations were set. Otherwise very similar to C
* Python - Easy, was definitely nice to prototype with and try out some of the ideas, it would have been a pain to do a lot of the investigations with other languages.
* F# - I don't like the syntax... Just saying. Tooling is pretty good, although doing stuff in CLI vs Visual Studio code was annoying.
* Go - Code was easy, was really impressed with it's speed once I got working on it. Still don't like some of the syntax rules and the package management. I think my rating would change if I were more familiar with writing Go. I can see why people like it.
* Zig - Just a proper slog. I do find the allocator / memory management paradigm interesting but you really have to learn the ins and outs of it. I really hate the slog with the version changes and a lot of the examples and docs are out of date. I guess this will change with a V1.0 release but there is no date on this.

## Languge agnostic things

### Devcontainers

I tried to set up devcontainers for the project so that it's easier to manage dependencies.

### Project organisation

I've tried to develop these examples with a heavy TDD approach with high levels of unit test coverage, each 'module' is it's own folder in the `src` tree and this contains all the associated source and includes (where appropriate) as well as the test code for each module.

I've created several entities that seem to serve most needs

* Seed - the value being transformed
* Seed Range - The start and size of seeds to be transformed
* Seed Map - A single map from source to target
* Seed Map Layer - A collection of Seed Maps that form one layer of transformation
* Seed Map Layers - The total collection of Seed Map Layers that make up the entire transformation

* part a - The solution for part a
* part b - The solution for part b

There are also constructs for; Seeds, Seed Ranges.

The general structure for each project will reflect this format. I've tried to separate the code into folders / files that match this structure, this way each part is logically separated from each other.

### Use of inline debuggers

I have tried to set up some debug targets in the `.vscode` folder for each project so that they can be reused inside the devcontainers

## C

I like C as a language, but OMG does it need some quality of life improvements. I tried to use just raw C with as few bells and whistles as possible but I didn't arbitrarily limit myself to just C99. If the C standards have been released then they are good enough for me.

I used this as an excuse to heavily use memory management as it's not something I'm very comfortable using, and I made sure to use 'valgrind' heavily during unit and system testing to make sure I validated the different pieces of code cleaned up their memory correctly.

I want to try and find a better method of unit testing the "memory safety".

### Structure

I've organised the projects into folders that contain each module, It does make the CMake files a little more frustrating as there are more lines to add in the include directives, but again there are ways to arrange this logically in the CMake to make maintenance easier and more visible.

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

Even though this requires more boilerplate code (especially for C, C++ and CUDA for the CMakeLists.txt) I prefer the segmentation of the project. It separates each of the components of the code into separate components and each folder is a more self-contained system. I've tried to mirorr that in each one of the repos.

### Toolchain frustrations

I've had to learn more about CMake for my current job as most of the build systems are done with CMake. It's good enough but it is very frustrating and longwinded.

### Unit testing

I used CUnit for the unit testing framework. It's pretty pleasant and easy to use, but requires a lot of manual wrangling in comparison to more modern languages (like Rust, go and zig). You have to maintain the make / cmake files which is just a PITA

## C++

I'm having to do more C++ for my job at the moment, and I'm finding it pretty pleasant as there are some quality of life improvements over C. 

After seeing the drastic difference in performance between C and C++ I had to use `gprof` to profile the code to see exactly what was taking the most time. I'm going to analyse this further in the [C++ Readme](../c++/README.md)

### GoogleTest

I used googletest for the unit testing and I'm not sure which one I prefer. I think that googletest has better support for mocking than Cunit, there is less boilerplate than in CUnit but I like the use of the different functions for assertions as it's a little clearer what is happening.

Suffers from the same problem as the C unit testing in that there is a fair amount of wrangling to do.

### HORRIFIC PERFORMANCE IMPLICATIONS in unoptimised code

I decided to use the `optional` keyword to return `nullptr` or a `uint64_t` value from the map seeds functions. The problem is that this cost a huge amount, almost doubling the execution time of the code. 

The use of vector iterators was also another serious performance hit which was hard to stomach in some cases. Especially when it's being executed millions of times. So don't use them for performance critical things.

### Optimisation

The problems with the previous section did seem to go away when I compiled with `-O3`. I want to spend some time researching this but don't have it at the moment.

## Cuda

This was my first foray into Cuda programming and it was a bit more pleasant than I'd anticipated although sharing memory between the CPU space and GPU space is something to consider carefully.

My solution 'works' but isn't very good and could definitely use some more work to make it good. I need to research how to leverage the blocks and threads more effectively to parallelise the computation and calculate the minimum on the fly better.

### Unit testing

I used CUnit for the main C code and this can be used to test the cuda code (grey box ish) by calling the `__global__` code but none of the `__device__` code could be tested without frustrating test harnesses.

## f#

This was an experiment to program specifically in a functional lanugage without using any imperative constructs, however it didn't work. For the sample it worked fine but when calculating the full values I ran into the bredth first search problem as it had to keep track of both the input vector and output vecfor which would have been 32Gb of memory

I'm not a fan of the syntax, and it's been a while since I wrote it so I didn't feel comfortable trying to quickly refactor it. I did want to try and get it to be completely functional by splitting the ranges into smaller numbers.

## go

Initial feel is that I don't like Go very much as the language seems very picky about what is 'allowed' and the conventions seem forced. 

I have managed to get a hold of the syntax rules and some other bits but I'm still not a huge fan. I really don't like the package management.

I like the unit testing conventions of keeping the unit tests close to the source it's testing. I could see myself becoming happier with it as a language in general, once I've got the syntax down I can imagine it being quite quick to develop with.

## Python

This was the first implementation which was quick and dirty. Worked well and was very useful for helping when I wanted to figure out the range based calculation. I really didn't want to have to do that in C.

The performance implications of using python were staggering. On the sample set of 10,000,000 initial values it took 618s. In comparison to Rust which when optimised took 0.6s. That 1000 times slower...... 

I do like python for a number of reasons, it's a quick language to develop in, and there are libraries for everything, but I think it has to be a serious consideration that the performance can be this bad. I think my solution is to try and write things in a way that means I can translate it easily to Rust (trying to write things functionally, reduce mutability)

## Rust

A lot smoother than I'd anticipated. The use of the rayon crate and the 'par_iter' function was beautiful. I like Rust a lot.

~~I don't like rust's import system. It's really clunky and very frustrating.~~ I take that back, it was a skill issue.

OH MY GOD the performance implications of using the release optimisations in rust! I turned up the optimisations to the max and turned off the integer overflow checking and it sped up the program by at least 10x.

## Zig 

I don't like zig very much. I dislike the build system as it's a bit difficult to fathom, and you have to define what modules are visible to other modules explicitly. I can see why this decision was chosen but  The syntax isn't bad and I like the use of the test code inside the module.

The low-level ness of the language seems to go overboard, I need to look up the purpose of the 'allocator' for things, it just seems to create a level of unnecessary control. I need to research this more.

I knew that I was stepping into a hornets nest by looking at a pre 1.0 release but the flux in the API is pretty frustrating. 
Advent of code 2023 Day 05 - Go
-------------------------------

# Overview

This wasn't unpleasant, but the language does a few things that I don't really like. Multiple return types are nice and the syntax is at a similar level to python so it's familiar enough.

My main issue was with the folder / code management and the general accessability of the codebase. I don't like the idiomatic code style and package structure, and the conventions for naming folders, imports and classes and I'm not a fan of the visibility of the function being based on the capitalisation of the first character.

# Build / Run

```
go build src/main.go
./main -i [path] -r [run]
```

Where [run] is:
* part_a
* part_b

# Future

## Concurrency

* https://goperf.dev/01-common-patterns/worker-pool/
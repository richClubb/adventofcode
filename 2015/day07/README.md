# AoC 2015 - Day 07

https://adventofcode.com/2015/day/7

# Status

* Part A - COMPLETE
* Part B - COMPLETE

# Notes

This was probably the first difficult one I found. It's not particularly hard but it's got some tricky frustrations and I don't think I did it very well, the code was overlong and a bit frustrating.

The general solution is that the operations were stored in a hashmap, and as they were "solved" by recursively searching in the map for the outputs their corresponding entry was reduced to just the value

E.g
```
5 -> a
10 -> b
c AND d -> e
g AND f -> c
15 -> f
1 -> d
8 -> g
c OR g -> h
```

`a`, `b`, `f`, `g` and `d` are just assignments so no operation needs to be done, as it gets to `e` it has to solve the function. So it takes `c` as the first input, finds the `c` entry in the hashmap, finds that `c = g AND f` it then finds `g` and `f`, calculates the result and then stores that result back in the hashmap for `c`. So it becomes

```
5 -> a
10 -> b
c AND d -> e
8 -> c
15 -> f
1 -> d
8 -> g
c OR g -> h
```

then does the same for `c and d` 

```
5 -> a
10 -> b
0 -> e
8 -> c
15 -> f
1 -> d
8 -> g
c OR g -> h
```
and then the same for `c OR g`

```
5 -> a
10 -> b
0 -> e
8 -> c
15 -> f
1 -> d
8 -> g
8 -> h
```

This minimises the number of operations that need to be done for the different calculations. Otherwise in the complex dataset it was just taking ages as there were potentially hundreds of calculations deep to keep iterating though.

I'm happy I solved it and the unit tests were invaluable to check my logic but I'm not happy as this seems like a massive amount of information and shouldn't have been that complex. I'm sure there is a nice way to tokenize it.

# Build / Run

## Build

```
cargo build [--release]
```

## Run

```
cargo run [path]
```

or

```
cd [build dir]
./day11 [path]
```

Part B uses the part_a file with a modification.

# Improvements / research

* Memoization / Caching of the functions
* tokenization
* using smart pointers?

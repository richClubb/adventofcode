Advent of code 2023 Day 05 - F#
-------------------------------

## Overview

This was probably my least favorite implementation. The syntax is not nice to work with and the learning curve was pretty massive. One of the nice things is that F# is not a purely functional language and you have got the break glass of imperative programming with mutable variables if you really need it. I'm writing this a long time after writing the code so I'm very rusty but the fact that I can't understand some of the pipe operators `||>` and `|>` at a glance is fairly telling.

I did have a big problem where the "functional" solution actually used a massive amount of memory, so much that my PC actually crashed the program as it was essentially doing a breadth first approach and storing all the values in RAM before finding the minimum of the output vector, this would consume around 16GB of RAM which was moe than my machine at the time had.

It does have some good sides, the runtime seems to automatically parallelise this to speed things up but as I haven't split up the ranges into a number to correspond to the number of cores then it doesn't optimise fully.

## To-do

* Command line parsing to accept the path and the run you want to perform
* Split the ranges into smaller chunks so that it can be processed functionally rather than imperatively
* Fill in test section

## Build / Run

From the `app_code` directory run

```
dotnet build
dotnet run
```

To change the path, alter the `path` variable in `./app_code/Program.fs`. It's unlikely that I'm going to spend the time to get command line parsing working for this so this is pretty much it. It runs both Part A and Part B every time.

## Test

```
dotnet test
```

There is currently an incorrect test but I am not going to fix it.
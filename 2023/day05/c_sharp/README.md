Advent of code 2023 Day 05 - C# 
-------------------------------

# Overview

This was fairly pleasant. I'm a fan of C# in general, it's not a bad language, a very good all-rounder and the package management is fairly nice. 

One big snag is that originally I tried to keep to the "everything is an object" mentality but I found implementing a "nullable" class was a bit of a PITA so I just settled to use the standard UInt64 with the "?" 

I didn't have to use any complex features or functions so I think the benefits of the language were not fully utilised. It was about twice as slow as C, C++, Rust and Zig but it's not that bad when you think it's not really designed as a "high performance" language.

# Build / Run

From the same directory as the readme

```
dotnet build
./app_code/bin/Debug/net9.0/app_code -c [type] --file [file] --run [run]
```

Where: 
* [type] is `Debug` or `Release`
* [file] is the path
* [run] is `part_a` or `part_b`.

# Test

Easy enough 

```
dotnet test
```

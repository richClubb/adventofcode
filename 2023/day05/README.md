# Day 5

## Languages

See [the languages page](./Languages.md) for more details on the implementations in each language.

* [C](./c/) - COMPLETE
* [C++](./c++/) - COMPLETE
* [Cuda]() - COMPLETE (ISH)
* [F#](./f_sharp/) - COMPLETE
* [Go](./go/) - Not going to do this
* [Python](./python/) - COMPLETE
* [Rust](./rust/) - COMPLETE
* [Zig](./zig/) - INCOMPLETE

## Overview

This has become a bit of a learning experiment for me. It is a cool problem that has a very simple soluttion but can be handled in a number of different ways and has a lot of cool methods to optimise the result.

[Original Page](https://adventofcode.com/2023/day/5) You'll need to be signed up for the advent of code to be able to view it. Each person gets their own input data so you can't just copy someone elses values. I've copied out the text into [problem.md](./docs/problem.md)

## Problem TLDR

You are given an initial value (a seed) and have to pass that value through a series of transformations to get a final value. You're looking for the minimum output from a set of inputs. Each layer will have a set of 'ranges' which map input values to output values, and if the input value doesn't fall into one of those ranges it will pass through unchanged.

```
      input val
          |
    -------------
    | map layer |
    -------------
          |
    output val (intermediate)
          |
         ...

         ...
          |
    -------------
    | map layer |
    -------------
          |
          |
     Final value
```

Initially in "Part A" of the problem you have 20 input values, and in "Part B" this increases to 2,000,000,000 input values. It's a classic O(n) scaling problem as the 'obvious' solution created in "Part A" doesn't scale well.

## Pseudo Solutions

Note: For all of these I'm ignoring the number of 'layers' or 'maps' in the O(n) calculation as the problem focuses on the number of input values being the major variable.

### "Obvious" depth first

1) Take an input value
2) Pass it through each layer
3) If the value is less than the current minimum, it becomes the new minimum
4) Repeat from Step 1 until no more input values remain
5) Output the minimum value

This has an O(n) time complexity where 'n' is the number of possible input values. For the sample data this is 4 for Part A and 27 for Part B, and for the full data 20 for part A and nearly 2,000,000,000 for part B. The space complexity is O(1) as you're only working on 1 value at a time. 

### Bredth first

1) For each value in the input, pass it through the first layer and calculate the output for that layer
2) Repeat 1) for each layer
3) Find minimum value in array
4) Output minimum value.

This has an O(n) time complexity where N is the number of possible input values. For the sample data this is 4 for Part A and 27 for Part B, and for the full data 20 for part A and nearly 2,000,000,000 for part B. The space complexity is O(n) where 'n' is the number of input values.

In practice this is not feasible on most systems as the RAM required is 2,000,000,000 * data type size. So using a uint64 that would be 16 Gb

### "Range" manipulation

1) For input range (for a single value the range is just the start value with a size of 1) calculate how the range "interacts" with the maps in each layer. This could result in up to 3 output maps. (See [range analysis notes](./Range%20Analysis%20Notes.pdf) for more details.
2) Repeat for each layer
3) Sort the ranges to find the range with the lowest start value
4) Output the lowest start value

This has an O(n) time and space complexity where 'n' is the number of 'input ranges' that you have. For Part A this is 4, and for Part B this is 2 using the sample data. Using the full data, Part A is 20 and part B is 10.

### Reverse search

1) Start from 0 and apply the transformations in reverse
2) See if the value is in the input set, if it is then output the initial value.
3) Increase the input value by 1 and repeat.

This has an O(n) time complexity, but is weird as there is the possibility it could take up to whatever the max value for the datatype is. E.g. for uint64 this would be 18,446,744,073,709,551,615.... So essentially infinite.

### Conculsion

We can see from this that the "Range" manipulation version is probably the fastest, but requires more code and calculation rather than the first 2, but it scales better depending on how large the "ranges" are. If the ranges are large then it's just 1 computation but if the ranges are small and there are large numbers of them then this scales just as badly as the depth first and bredth first approaches.

### Optimisations (language agnostic)

These optimisations are language agnostic methods of speeding up the approach.

#### Flattening

In both the sample data and the full data we have 7 layers of transformations. In all 3 solutions we've outlined we have to go through all 7 layers for each input value. We could perform a flattening of each layer using the same technique in the "range" manipulation, this could have a significant increase in speed as we'd only have to go through 1 layer each time

#### Sorting

The mappings are not sorted in the layers, so we could sort these such that we could find where the input value 'falls' in the layer and exit quicker rather than having to process each layer in it's entirity. This is especially useful if the value doesn't map as otherwise it would have to go through each map in each layer before it can exit. This would not have as much of an optimisation as the flattening, but we could sort the flattened layer and increase the speed as well.

This is not 'necessary' for the range manipulation approach but certainly helps.

### Parallel processing

If we arrange the data we can process the input in parallel across many processors. A problem occurs when thinking about how to calculate the minimum of the values. In my approaches the "ranges" are a logical method of organising each thread, so that each thread takes one range and calculates the minimum value, then you only have to find the minimum of the minumum of each range.

#### Splitting the ranges

The ranges are not evenly sized, so splitting the ranges into more even sets would speed things up as each processor would have a more even dataset to process.
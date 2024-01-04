Notes on solving Advent of Code - day 5
---------------------------------------

This is a case study on solving Day 5 of the avent of code.

# Overall Problem

You have to translate in input value (seed) through a series of arbitrary mappings to an output value.

There are 7 layers of mappings, each "layer" can have multiple maps.

You have to find the lowest value of the output for a particular input value.

# Part A

This starts out saying that the seed values are as they are written in the file.

seeds: 28965817 302170009 1752849261 48290258 804904201 243492043 2150339939 385349830 1267802202 350474859 2566296746 17565716 3543571814 291402104 447111316 279196488 3227221259 47952959 1828835733 9607836

Which is 20 values.

The maps are arranged as 3 values; Output Start, Input Start, Size

E.g.

1 10 5

would map 10, 11, 12, 13, 14 to 1, 2, 3, 4, 5

The most obvious solution here is to just do a depth first approach and put each seed value through the mappings and get an output value. Compare that against the last and see if it is smaller, if so then it becomes the smallest known value and we continue.

Another way to do it would be a bredth first approach by putting all the seeds through each layer and then finding the min value.

Both approaches take negligable time on a normal computer and is done in milliseconds. Coding the depth first took about 30 - 45 minutes from start to finish in the Midlands Engineering meeting.

# Part B

This throws a spanner in the works by massively increasing the number of seeds to search. They say that the "seeds" are actually pairs of numbers. with the first value being the start and the next value being the size.

seeds: 28965817 302170009 1752849261 48290258 804904201 243492043 2150339939 385349830 1267802202 350474859 2566296746 17565716 3543571814 291402104 447111316 279196488 3227221259 47952959 1828835733 9607836

becomes 
 
     start      size
  28965817 302170009 
1752849261  48290258 
 804904201 243492043 
2150339939 385349830 
1267802202 350474859 
2566296746  17565716 
3543571814 291402104 
 447111316 279196488 
3227221259  47952959 
1828835733   9607836

This ends up with about 2 billion seeds (1,975,502,102).

## Bredth First approach

This immediately goes out the window. 2 billion times 8 as some of the values are larger than a Uint 32 can store so we have to use Uint64. This ends up with 8*2,000,000,000 = 16,000,000,000 bytes which is around 15 gig of memory to store each layer.

## Overall problems

This change immediately throws a spanner in the works. Our previous approach works by processing each seed and checking it is smaller than the last. This means you have to compute 2 billion entries. On my surface book it took 5 seconds to process 100000 that means it would take about 27 hours to complete.

## Potential Solutions

### Parallel Computation

It is very easy to set this up as a parallel processing job. On the sample data this sped up the results significantly. Each process takes slightly longer but they can work in parallel. d

--------------------------------------------------------------------------------------

def part_b_forward(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)
    min_loc = 10**30

    total_seeds = 0
    for _, seed_length in seed_pairs:
        total_seeds += seed_length

    for seed_start, seed_length in seed_pairs:
        for seed in range(seed_start, seed_start + seed_length):
            loc = find_location(seed, maps)
            if loc < min_loc:
                min_loc = loc

    return min_loc

--------------------------------------------------------------------------------------

def part_b_forward_multiprocess(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)
    pool_arguments = []

    for seed_pair in seed_pairs:
        pool_arguments.append((seed_pair, maps))

    with Pool(4) as p:
        results = p.map(find_lowest_location_mp, pool_arguments)

    return min(results)
    
def find_lowest_location_mp(arguments):
    seed_start, seed_length = arguments[0]
    maps = arguments[1]

    min_loc = 10**30

    for seed in range(seed_start, seed_start + seed_length):
        loc = find_location(seed, maps)
        if loc < min_loc:
            min_loc = loc

    return min_loc

--------------------------------------------------------------------------------------

This took 10 - 15 minutes to implement and sped up the procss greatly. Each process takes 7 seconds to process 100000 entries, and I had 4 in parallel. Which takes it down to 9 hours

This idea can be taken to massive extremes. One comment on reddit said that a brute force approach when coded in CUDA to run on an Nvidia 3080 took 0.8s but this relies on specialist hardware. This is using C so it's not a complete apples to apples comparison as we'll see with Rust.

### Using a diferent language

My solutions are are all based on Python. But using Rust to solve the same problem in the same way took 7 minutes single threaded (on a different PC which equated to 100x improvement as their brute force forward processing took 700 minutes), so if you take a similar parallel so in parallel it would take about 3 minutes.

ChatGPT proved to be really useful here as it was able to translate the code into Rust really quickly so this could be a very useful tool to leverage. If you need a particular part of your code to be performant but you don't want to rewrite the whole thing then you could carve out part of the code, get ChatGPT to translate it into Rust or C and then call into it from the main program.

There are a number of languages that can be called from Python so this is a useful string to have in the bow. There is a python CUDA implementation but I have not yet figured out how to get it to work on my surface book but I'm definitely going to try. 

### Reverse mapping.

Another approach is working backwards. If you start from 0 (lowest possible value if they are all positive integers) you can work backwards through the mapping end see if the resulting seed is in one of the ranges.

This took about an hour to code up, and took about 70 minutes to solve the problem. It has certain advantages over the "forward" processing as you don't have to search the whole serach space to find a result.

There is a problem with this approach. You're starting from 0 but due to the problem you have no idea where might be a better place to start. So you might end up having millions of computations before you end up with a result. There could be a way to split this up into a "search" process and a brute force process but this increase the code complexity and doesn't guarantee it will be faster in any solution.

This could also be parallelised by taking blocks of numbers E.g. 0-99999, 100000 - 199999 ... and processing them in parallel and checking the lowest value returned from the blocks. This is easy enough to code and could speed up the process significantly. It took about 30 minutes to code this up and completed in 40 minutes. so about 1.75 times the speed of the original reverse mapping.

### Working on the Ranges of seeds, rather than the individual seeds

Another idea is to translate the range, rather than the individual seeds. This is more complex to think about but you'd end up with 10s or 100s of computations rather than billions.

     start      size
  28965817 302170009 
1752849261  48290258 
 804904201 243492043 
2150339939 385349830 
1267802202 350474859 
2566296746  17565716 
3543571814 291402104 
 447111316 279196488 
3227221259  47952959 
1828835733   9607836

Becomes

     Start         End
  28965817   331135826
1752849261  1801139519
 804904201  1048396244
2150339939  2535689769
1267802202  1618277061
2566296746  2583862462
3543571814  3834973918
 447111316   726307804
3227221259  3275174218
1828835733  1838443569

You can then figure out how each range would pass through each map

E.g.

Seed: 1 10
Map   20 5 3

Would result in: (1 4) (20 22) (8 10)

This is harder to figure out but computationally you end up significantly less to do. There were 11 cases where the seed interacts (and 2 where it doesn't) with the map. I could possibly reduce these with a little more thinking but this would be a sub optimisation. Look at 'Range Analysis Notes.pdf' for my hand written notes.

I decided to code the cases in cascading functions as it seemed nicer than having them as a number of if else if statements. Which I thought was easier to read and debug. You could unit test each case individually if you wanted rather than having to unit test the entire if-else statement.

This took about a day to; think, code and troubleshoot but the computation takes 0.1 seconds on my computer, which is a nearly 1000000x improvement in performance, it doesn't need to be parallelised (but could easily if you took each seed range and processed it in parallel) and could probably be made even faster by processing it in Rust or C. So thinking about it, if I'd set off my original code running I'd have written this new version and got a result before the original completed.

The code is significantly longer and more complex. I've got helper functions to make the arrays easier to read.

# Conclusion

* Your initial design might have been good for where you started, but it might not end up being good for where you are going
* Don't be afraid to kill your darlings.
* Think about your objectives. If you wanted to "complete the advent of code" it might just be fine to leave it running for a day and get the answer. I viewed this as a journey to solve the problem and find interesting solutions, and I had a great time figuring it out. I got to find out about Python CUDA and experiment and it was a good excuse to do that. But I did not complete the Advent of Code.



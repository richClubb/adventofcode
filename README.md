# Advent of Code

My repository for the advent of code solutions

# Total Solved

* 52 / 524 (9.9%)

# Breakdown by Year

* [2015](./2015/) - 34 / 50 - Challenge - just use Rust.
* [2016](./2016/) -  0 / 50
* [2017](./2017/) -  0 / 50
* [2018](./2018/) -  0 / 50
* [2019](./2019/) -  0 / 50
* [2020](./2020/) -  0 / 50
* [2021](./2021/) -  8 / 50 - Not currently documented here, need to re-create
* [2022](./2022/) -  2 / 50 - Not currently documented here, need to re-create
* [2023](./2023/) -  6 / 50 - Challenge - Every problem in Rust, Python and Zig. I didn't get far with that challenge.
* [2024](./2024/) -  0 / 50
* [2025](./2025/) -  2 / 50

# Notes

## 2023 Day 05

Look at the [2023 day 05](./2023/day05/) I did a big study on this one as it was super interesting for me.

I programmed it in a bunch of languages with multiple different ways to solve the problem. Was a load of fun and really enjoyed doing this even if it absorbed about 2 months of my evenings.

## Solutions only!

Respecting the [wishes of the author of AoC](https://adventofcode.com/2015/about#faq_copying) I've copied all the problems and the inputs but I've encrypted them so as not to distribute them. This is mostly so I can keep them just in case he takes the site offline.

Always respect the wishes of the creators. If you don't like it, go and do something else.

# Encrpytion / Decryption Notes

This is mostly for me just in case I forget.

## Encrypt

```
tar -cvzf - problem_solution_input/ | gpg -e -r [id] > problem_solution_input.tar.gz.gpg
```

## Decrypt

```
gpg -d problem_solution_input.tar.gz.gpg | tar -xvzf -
```

# Challenge Ideas

* Only use functional programming techniques
* Do 1 year using the 25 most popular programming languages, one for each day
    * This looks awful. Cobol, SQL and Scratch... Ada?
* Do a year using just Rust - In progress
* Do a year using a microcontroller - Limited RAM and processor capacity
    * This could also be a container with crippled capabilities just to simulate
    * "every problem has a solution that completes in at most 15 seconds on ten-year-old hardware" is taken from the FAQ


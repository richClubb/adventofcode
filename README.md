# Advent of Code

My repository for the advent of code solutions

# Total Solved

* 76 / 524 (14.5%)

# Breakdown by Year

* [2015](./2015/) - 50 / 50 - Challenge - just use Rust.
* [2016](./2016/) -  8 / 50 - Challenge - Alternate between C, Go, Rust, Python and Zig. Don't use same language twice in a row. Should end up doing each language 5 times.
* [2017](./2017/) -  0 / 50
* [2018](./2018/) -  0 / 50
* [2019](./2019/) -  0 / 50
* [2020](./2020/) -  0 / 50
* [2021](./2021/) -  8 / 50 - Not currently documented here, need to re-create
* [2022](./2022/) -  2 / 50 - Not currently documented here, need to re-create
* [2023](./2023/) -  6 / 50 - Challenge - Every problem in Rust, Python and Zig. I didn't get far with that challenge as it's just a lot of work. Instead, try every problem in Rust, Python or Zig and you have to do all three before you can re-use one and you can't use the same one twice in a row. E.g. Rust, Zig, Python, Zig, Python, Rust.
* [2024](./2024/) -  0 / 50
* [2025](./2025/) -  2 / 50

# Notes

## 2023 Day 05

Look at the [2023 day 05](./2023/day05/) I did a big study on this one as it was super interesting for me.

I programmed it in a bunch of languages with multiple different ways to solve the problem. Was a load of fun and really enjoyed doing this even if it absorbed about 2 months of my evenings.

## Solutions only!

Respecting the [wishes of the author of AoC](https://adventofcode.com/2015/about#faq_copying) I've copied all the problems and the inputs but I've encrypted them so as not to distribute them. This is mostly so I can keep them just in case he takes the site offline.

Always respect the wishes of the creators. If you don't like it, go and do something else.

#3 Encrpytion / Decryption Notes

This is mostly for me just in case I forget.

### Encrypt

```
tar -cvzf - problem_solution_input/ | gpg -e -r [id] > problem_solution_input.tar.gz.gpg
```

### Decrypt

```
gpg -d problem_solution_input.tar.gz.gpg | tar -xvzf -
```

# Challenge Ideas

* Only use functional programming techniques
* Do 1 year using the 25 most popular programming languages, one for each day
    * This looks awful. Cobol, SQL and Scratch... Ada?
    * Can I get toolchains for each of these.
    * Will have to re-do the devcontainer framework.
* Do a year using just Rust - In progress (2015)
* Do a year using a microcontroller - Limited RAM and processor capacity
    * This could also be a container with crippled capabilities just to simulate
    * "every problem has a solution that completes in at most 15 seconds on ten-year-old hardware" is taken from the FAQ


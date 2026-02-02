# AoC 2015 - Day 22

https://adventofcode.com/2015/day/22

# Status

* Part A - COMPLETE
* Part B - COMPLETE

# Build / Run

## Build

```
cargo build [--release]
```

## Run

```
cargo run [run]
```

or

```
cd [build dir]
./day22 [run]
```

where `[run]` is:
* part_a
* part_b

# Notes

This is different from the last problem as we have incomplete information.

Can this be done as a decision tree?

Depth first search?

The problem says that a spell with an effect doesn't stack E.g. only one poison, shield, recharge can be active at a time.

There should always be an exit condition, either the boss dies or the player dies.

E.g.

* if you just cast magic missile, after 9 rounds the player would run out of mana but would only have done 36 damage.
  * The boss would also kill you after 5 rounds
* if you cast poison, you can't cast poison for another 6 turns, so your choices are MM, Drain, Shield, Recharge.
  * if you cast poison, shield, recharge, you could only cast MM or Drain till the effects run out.
* The thought is that if I do a depth first, and keep track of the total mana cost.
  Each iteration I can keep going. 
  * How do I keep track of the effects?

# Improvements

I'm not really happy with the solution, I want to try a different approach at some point
#!/bin/env python3

import functools
import math

input = [
    (28965817, 302170009),
    (1752849261, 48290258), 
    (804904201, 243492043),
    (2150339939, 385349830),
    (1267802202, 350474859),
    (2566296746, 17565716),
    (3543571814, 291402104),
    (447111316 ,279196488),
    (3227221259, 47952959),
    (1828835733, 9607836)
]

def split_range(start, size, ideal_size):
    if size <= ideal_size:
        new_list = [(start,size)]
        print("smaller {} {}".format(size/ideal_size, new_list))
        return new_list
    
    if size / ideal_size <= 2:
        print("greater but less than or equal to 2 {}".format(size/ideal_size))
        new_size = size / 2
        new_list = [(start, new_size), (start+new_size, new_size)]
        print("greater but less than or equal to 2 {} {}".format(size/ideal_size, new_list))
        return new_list

    if size / ideal_size > 2:
        number = int(math.floor(size / ideal_size))
        new_size = int(math.ceil(size / number))
        print("greater than 2 {} {}".format(number, new_size))

        new_list = []
        remaining = size
        new_start = start
        while remaining > 0:
            if remaining > new_size:
                new_list.append((new_start, new_size))
                new_start += new_size
                remaining -= new_size
            else:
                new_list.append((new_start, remaining))
                remaining -= remaining

        print("greater than 2 {} {} {}".format(number, new_size, new_list))
        return new_list
    pass

def main():

    input_sorted = sorted(input, key=lambda entry: entry[1])

    processors = 28

    ideal_size = round(functools.reduce(lambda x, entry: x + entry[1], input_sorted, 0) / processors)
    
    new_input = []
    for entry in input_sorted:
        new_input += (split_range(entry[0], entry[1], ideal_size))

    print(new_input)
    print(len(new_input))

if __name__ == "__main__":
    main()
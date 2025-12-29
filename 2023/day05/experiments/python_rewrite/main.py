
from part_a.part_a import part_a
from part_b.part_b import part_b_forward, part_b_inverse, part_b_forward_parallel

import argparse
import os

RUNS = ["part_a", "part_b_forward", "part_b_inverse", "part_b_forward_parallel"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file_path")
    parser.add_argument("run", choices=RUNS)

    args = parser.parse_args()

    if os.path.exists(args.input_file_path) is False:
        print("Missing input file")
        exit()

    print("Advent of code 2023 day 5")

    if args.run == "part_a":
        print(f"part a (forward depth first): {part_a(args.input_file_path)}")
    elif args.run == "part_b_forward":
        print(f"part b (forward depth first): {part_b_forward(args.input_file_path)}")
    elif args.run == "part_b_forward_parallel":
        print(f"part b (forward depth first): {part_b_forward_parallel(args.input_file_path)}")
    elif args.run == "part_b_inverse":
        print(f"part b (inverse depth first): {part_b_inverse(args.input_file_path)}")
    
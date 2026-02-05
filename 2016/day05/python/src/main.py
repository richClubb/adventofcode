
#!/bin/env python3

import argparse

if __name__ == "__main__":
    pass

    parser = argparse.ArgumentParser(
        prog='day05',
        description='Advent of code 2016 Day 06',
        epilog='Merry Christmas (in February)'
    )

    parser.add_argument("-i", "--input")
    parser.add_argument("-r", "--run")

    args = parser.parse_args()
    print(args.input, args.run)
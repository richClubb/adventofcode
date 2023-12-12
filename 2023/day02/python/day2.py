#!/bin/env python3
import argparse
import os
import re


def check_red(input_round):
    if match:= re.search("([0-9]{1,})(red)", input_round):
        if int(match.group(1)) > 12:
            return False
    return True
        
def check_green(input_round):
    if match:= re.search("([0-9]{1,})(green)", input_round):
        if int(match.group(1)) > 13:
            return False
    return True

def check_blue(input_round):
    if match:= re.search("([0-9]{1,})(blue)", input_round):
        if int(match.group(1)) > 14:
            return False
    return True


def is_round_possible(input_round):
    if len(list(filter(lambda x:(check_red(x)==False) or (check_green(x)==False or (check_blue(x)==False)), input_round.split(",")))) > 0:
        return False
    return True


def is_game_possible(input_game_string):
    if len(list(filter(lambda x:is_round_possible(x)==False,input_game_string.split(";")))) > 0:
        return False
    return True


def part_a(input_file_path):

    total = 0

    with open(input_file_path) as file:
        # for line in file.readlines():
        #     game_number, game_string = line.strip().replace(" ", "").replace("Game","").split(":")
        #     if is_game_possible(game_string) == True:
        #         total += int(game_number)
        list(filter(lambda x, y:is_game_possible(y)), line.strip().replace(" ", "").replace("Game","").split(":"))


    return total
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file_path")

    args = parser.parse_args()

    if os.path.exists(args.input_file_path) is False:
        print("Missing input file")
        exit()

    print(part_a(args.input_file_path))


#!/bin/env python3

import random

LANGUAGES = ["Python", "Rust", "C", "Go", "Zig"]

def main():

    languages_to_use = LANGUAGES.copy()
    random.shuffle(languages_to_use)
    for index in range(0, 25):
        
        language = languages_to_use.pop()
        print(f"{index+1} {language}")

        if len(languages_to_use) == 0:
            languages_to_use = LANGUAGES.copy()
            
            while True:
                random.shuffle(languages_to_use)
                if index > 0:
                    if languages_to_use[0] == language:
                        print("Shuffling")
                        random.shuffle(languages_to_use)
                    else:
                        break

if __name__ == "__main__":

    main()
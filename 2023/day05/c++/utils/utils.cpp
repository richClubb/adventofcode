#include "utils.h"

#include <stdint.h>

#include <bits/stdc++.h>
#include <vector>
#include <string>

std::vector<uint32_t> extract_numbers(std::string input_string)
{
    std::stringstream ss(input_string);

    std::vector<uint32_t> numbers;

    std::string token;
    while(getline(ss, token, ' '))
    {
        numbers.push_back(std::stol(token, 0, 10));
    }

    return numbers;
}
#include "utils.h"

#include <stdint.h>

#include <bits/stdc++.h>
#include <vector>
#include <string>


std::vector<uint64_t> get_seeds(std::string input_string)
{
    int32_t pos = input_string.find("seeds: ");

    std::string number_string;
    if (pos == 0)
    {
        number_string = input_string.substr(7);
    }

    return extract_numbers(number_string);
}


std::vector<uint64_t> extract_numbers(std::string input_string)
{
    std::stringstream ss(input_string);

    std::vector<uint64_t> numbers;

    std::string token;
    while(getline(ss, token, ' '))
    {
        numbers.push_back(std::stol(token, 0, 10));
    }

    return numbers;
}
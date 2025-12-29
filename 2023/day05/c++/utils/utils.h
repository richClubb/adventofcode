#ifndef __UTILS_H__

#define __UTILS_H__

#include <stdint.h>
#include <vector>
#include <string>

std::vector<uint64_t> get_seeds(std::string input_string);

std::vector<uint64_t> extract_numbers(std::string);

#endif
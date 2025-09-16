
#include <stdint.h>

#include "config.h"

#include <iostream>
#include <fstream>

uint32_t part_a(const CONFIG &config)
{
    std::ifstream input_file(config.input_file_path);

    std::string line;
    while(getline(input_file, line))
    {
        if( line.length() < 2 )
        {
            continue;
        }

        if( line.find("seeds: ") != std::string::npos)
        {
            printf("Found seeds line\n");
            continue;
        }

        if (line.find(":") != std::string::npos)
        {
            printf("Found seed map layer def line\n");
            continue;
        }

        printf("Line is a seed map\n");
    }

    return 1;
}
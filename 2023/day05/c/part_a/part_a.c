#ifdef TEST
    #include "test_header_part_a.h"
#else
    #include "part_a.h"
#endif

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <regex.h>


#include "config.h"
#include "seed_map_layer.h"

// manually interprets the line
long *get_seeds(char *line, unsigned int *num_seeds)
{
    // seed line always starts with 'seeds: ' so atrip that out
    unsigned int seeds_substring_len = strlen(line) - sizeof("seeds: ");
    char *seeds_substring = (char *)calloc(seeds_substring_len + 1, sizeof(char));

    // if this fails then we can't continue
    assert(seeds_substring != NULL);
    
    // copy in the substring
    strncpy(seeds_substring, line + 7, seeds_substring_len);

    // allocate the inital seeds array
    long *seeds = (long *)calloc(1, sizeof(long));
    unsigned int seeds_index = 0;

    // iterate over the seeds to get the values
    char* token = strtok(seeds_substring, " ");
    while (token != NULL) {
        char* endptr = NULL;

        seeds[seeds_index] = strtol(token, &endptr, 10);
        seeds_index += 1;
        seeds = (long *)realloc(seeds, sizeof(long) * seeds_index + 1);

        token = strtok(NULL, " ");
    }

    // free seeds_substring as we no longer need it.
    free(seeds_substring);

    // update the num seeds and return the seeds list
    *num_seeds = seeds_index;
    return seeds;
}

// uses a regex
SEED_MAP *get_seed_map(char *line)
{
    long *map_values = (long *)calloc(3, sizeof(long));

    free(map_values);

    return NULL;
}

char *get_seed_map_layer_name(char *line)
{

}

void part_a(const CONFIG *config)
{

    FILE *input_file = fopen(config->input_file_path, "r");

    // bomb out if the file is NULL
    assert(input_file != NULL);

    long *seeds;
    unsigned int num_seeds;
    // check for memory usage
    SEED_MAP_LAYER **seed_map_layers = (SEED_MAP_LAYER **)malloc(sizeof(SEED_MAP_LAYER *) * 1);
    unsigned int seed_map_layers_index = 0;

    char line[256];
    for(
        char line[256]; 
        fgets(line, sizeof(line), input_file) != NULL;
    ) 
    {
        SEED_MAP_LAYER *curr_layer;
        if ( strlen(line) == 1 )
        {
            printf("Found empty line\n");
            continue;
        }

        if ( strstr(line, "seeds:") != NULL )
        {
            printf("Found seeds line\n");
            seeds = get_seeds(line, &num_seeds);
            continue;
        }

        if ( strstr(line, ":") != NULL )
        {
            printf("Found seed map layer definition\n");
            // memory check
            seed_map_layers = (SEED_MAP_LAYER **)realloc(seed_map_layers, sizeof(SEED_MAP_LAYER *) * seed_map_layers_index + 1);
            seed_map_layers[seed_map_layers_index] = (SEED_MAP_LAYER *)malloc(sizeof(SEED_MAP_LAYER));

            curr_layer = seed_map_layers[seed_map_layers_index];
            
            seed_map_layer_init(curr_layer);
            continue;
        }

        // if nothing else then it must be a seed map line
        printf("found seed map line\n");
    }

    // cleanup
    for(
        unsigned int index = 0;
        index < seed_map_layers_index;
        index++
    )
    {
        free(seed_map_layers[index]);
    }

    free(seed_map_layers);

    fclose(input_file);
    
    if(seeds)
    {
        free(seeds);
    }
}
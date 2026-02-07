#include "part_a.h"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "config.h"

/* ----------------------------------- */

typedef struct instruction_t {
    char direction;
    uint8_t distance;
} INSTRUCTION;

typedef struct position_t {
    int64_t x_pos;
    int64_t y_pos;
} POSITION;

typedef enum direction_t {
  NORTH,
  SOUTH,
  EAST,
  WEST,
} DIRECTION;

DIRECTION next_direction(DIRECTION origin, char turning) {

    switch (origin) {
        case NORTH:
            if      (turning == 'R') { return EAST; }
            else if (turning == 'L') { return WEST; }
            break;
        case SOUTH:
            if      (turning == 'R') { return WEST; }
            else if (turning == 'L') { return EAST; }
            break;
        case EAST:
            if      (turning == 'R') { return SOUTH; }
            else if (turning == 'L') { return NORTH; }
            break;
        case WEST:
            if      (turning == 'R') { return NORTH; }
            else if (turning == 'L') { return SOUTH; }
            break;
    }

    return NORTH;
}

INSTRUCTION* injest_file(const char *file_path, uint64_t* num_instructions) {

    FILE* input_file = fopen(file_path, "r");

    uint64_t instructions_buffer_size = 10;
    INSTRUCTION* instructions = calloc(sizeof(INSTRUCTION), instructions_buffer_size);

    uint64_t instruction_index = 0;

    char curr_char;

    char direction;
    char distance_buffer[10];
    uint8_t distance_buffer_index = 0;

    memset(distance_buffer, '\0', 10);

    while ((curr_char = (char)fgetc(input_file)) != EOF) {
        
        if (
            ('L' == curr_char) || 
            ('R' == curr_char)
        ) {
            direction = curr_char;
            continue;
        }
        else if (
            ('0' <= curr_char) && 
            ('9' >= curr_char)
        ) {
            distance_buffer[distance_buffer_index] = curr_char;
            distance_buffer_index += 1;
        }
        else if (',' == curr_char) {

            INSTRUCTION* curr_instruction = &instructions[instruction_index];

            curr_instruction->direction = direction;
            curr_instruction->distance = (uint64_t)atoi(distance_buffer);
            instruction_index += 1;

            if (instruction_index == instructions_buffer_size) {
                instructions_buffer_size = instructions_buffer_size * 2;
                instructions = realloc(instructions, instructions_buffer_size * sizeof(INSTRUCTION)); 
            }

            // clear temp buffers
            memset(distance_buffer, '\0', 10);
            distance_buffer_index = 0;
            direction = '\0';
        }
        else if (' ' == curr_char) {
            continue;
        }
    }
    // do last char fill in as we've hit the EOF
    if ('\0' != direction) 
    {
        INSTRUCTION* curr_instruction = &instructions[instruction_index];
        
        curr_instruction->direction = direction;
        curr_instruction->distance = (uint64_t)atoi(distance_buffer);
        instruction_index += 1;
    }

    *num_instructions = instruction_index;

    fclose(input_file);

    return instructions;
}

uint64_t part_a(const CONFIG *config)
{
    
    uint64_t num_instructions = 0;

    INSTRUCTION* instructions = injest_file(config->input_file_path, &num_instructions);

    printf("Num instruction %lu\n", num_instructions);

    // hardcoding this to 10000 as I CBA to figure out the dynamic resizing condition for this
    POSITION positions_visited[10000];
    uint64_t positions_visited_index = 1;

    int64_t x_pos = 0;
    int64_t y_pos = 0;

    DIRECTION curr_direction = NORTH;

    positions_visited[0].x_pos = 0;
    positions_visited[0].y_pos = 0;

    for (uint64_t index = 0; index < num_instructions; index++) {

        curr_direction = next_direction(curr_direction, instructions[index].direction);

        switch (curr_direction) {
            case NORTH:
                for (uint64_t index_int = 0; index_int < instructions[index].distance; index_int++){
                    positions_visited[positions_visited_index].x_pos = x_pos;
                    positions_visited[positions_visited_index].y_pos = ++y_pos;
                    positions_visited_index += 1;
                }
                break;
            case SOUTH:
                for (uint64_t index_int = 0; index_int < instructions[index].distance; index_int++)
                {
                    positions_visited[positions_visited_index].x_pos = x_pos;
                    positions_visited[positions_visited_index].y_pos = --y_pos;
                    positions_visited_index += 1;
                }
                break;
            case EAST:
                for (uint64_t index_int = 0; index_int < instructions[index].distance; index_int++){
                    positions_visited[positions_visited_index].x_pos = ++x_pos;
                    positions_visited[positions_visited_index].y_pos = y_pos;
                    positions_visited_index += 1;
                }
                break;
            case WEST:
                for (uint64_t index_int = 0; index_int < instructions[index].distance; index_int++)
                {
                    positions_visited[positions_visited_index].x_pos = --x_pos;
                    positions_visited[positions_visited_index].y_pos = y_pos;
                    positions_visited_index += 1;
    
                }
                break;
        }
    }

    for (uint64_t search_index_1 = 0; search_index_1 < (positions_visited_index - 1); search_index_1++ ) {
      
        bool completed = false;
        for(uint64_t search_index_2 = search_index_1 + 1; search_index_2 < positions_visited_index; search_index_2++ )
        {
            int64_t search_x_pos = positions_visited[search_index_1].x_pos;
            int64_t search_y_pos = positions_visited[search_index_1].y_pos;

            if (
                (search_x_pos == positions_visited[search_index_2].x_pos) &&
                (search_y_pos == positions_visited[search_index_2].y_pos)
            )
            {
                printf("Already visited (%ld, %ld) %ld\n", search_x_pos, search_y_pos, labs(search_x_pos) + labs(search_y_pos));
                completed = true;
                break;
            }
        }

        if (completed) {
            break;
        }
    }

    printf("Pos (%ld, %ld) distance: %ld\n", x_pos, y_pos, labs(x_pos) + labs(y_pos));

    return 0;
}
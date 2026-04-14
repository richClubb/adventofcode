#include "part_a.h"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "config.h"

typedef enum instruction_type_t {
    RECT,
    ROTATE
} INSTRUCTION_TYPE;

typedef enum direction_t {
    X_DIR,
    Y_DIR
} DIRECTION;

typedef struct instruction_t {
    INSTRUCTION_TYPE type;
    uint32_t x_size;
    uint32_t y_size;
    DIRECTION dir;
    uint32_t spaces;
} INSTRUCTION;


/* ----------------------------------- */

INSTRUCTION* injest_file(const char* input_file, uint32_t* num_inputs) {
    

}

uint64_t part_a(const CONFIG* config)
{

    FILE* input_file = fopen(config->input_file_path, "r");

    char buffer[1024];
    memset(buffer, '\0', 1024);

    while (fgets(buffer, sizeof(buffer), input_file)) {
        // Print each line to the standard output.
        printf("%s", buffer);
    }

    return 0;
}
#include "utils.cuh"

#include <assert.h>
#include <stdint.h>

uint64_t *extract_number_list(const char *number_string, uint64_t *length)
{
    char *mut_line = (char *)calloc(strlen(number_string), sizeof(char));
    memcpy(mut_line, number_string, sizeof(char) * strlen(number_string));

    uint64_t *numbers = (uint64_t *)calloc(1, sizeof(uint64_t));
    assert(numbers != NULL);

    uint64_t numbers_index = 0;

    char* token = strtok(mut_line, " ");
    assert(token != NULL);

    while (token != NULL) {
        char* endptr = NULL;

        numbers[numbers_index] = (uint64_t)strtol(token, &endptr, 10);
        ++numbers_index;
        // error checking
        numbers = (uint64_t *)realloc(numbers, sizeof(uint64_t) * (numbers_index + 1));

        token = strtok(NULL, " ");
    }

    free(mut_line);

    *length = numbers_index;
    return numbers;
}
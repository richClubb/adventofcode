#ifndef __UTILS_H__

#define __UTILS_H__

unsigned long *get_seeds(const char *line, unsigned int *num_seeds);

unsigned long *extract_number_list(const char *number_string, unsigned int *length);

#endif
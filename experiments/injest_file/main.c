#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void injest_file(uint64_t **seeds, uint64_t *num_seeds, uint64_t ***layers, uint64_t *num_layers)
{
    *seeds = (uint64_t *)calloc(4, sizeof(uint64_t));

    (*seeds)[0] = 10;
    (*seeds)[1] = 20;
    (*seeds)[2] = 30;
    (*seeds)[3] = 40;

    *num_seeds = 4;

    *layers = (uint64_t **)calloc(5, sizeof(uint64_t *));

    (*layers)[0] = (uint64_t *)calloc(5, sizeof(uint64_t));
    (*layers)[1] = (uint64_t *)calloc(5, sizeof(uint64_t));
    (*layers)[2] = (uint64_t *)calloc(5, sizeof(uint64_t));
    (*layers)[3] = (uint64_t *)calloc(5, sizeof(uint64_t));
    (*layers)[4] = (uint64_t *)calloc(5, sizeof(uint64_t));
    *num_layers = 5;
}



int main(int argc, char **argv)
{

    uint64_t *seeds;
    uint64_t num_seeds;

    uint64_t **layers;
    uint64_t num_layers;



    injest_file(&seeds, &num_seeds, &layers, &num_layers);

    for(uint64_t index = 0; index < num_seeds; index++)
    {
        printf("Seed: %lu\n", seeds[index]);
    }

    for(uint64_t index = 0; index < num_layers; index++)
    {
        printf("layer: %lu\n", index);
        uint64_t *layer = layers[index];
        for(uint64_t index_2 = 0; index_2 < 5; index_2++)
        {
            printf("  Entry %lu: %lu\n", index_2, layer[index_2]);
        }
    }

    return 1;
}
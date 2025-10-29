#include "part_b.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

#include "config.h"
#include "seed_map_layer.h"
#include "seed_range.h"
#include "utils.h"

#define DEVICE CL_DEVICE_TYPE_DEFAULT

const char *KernelSource_part_b = "\n" \
"typedef unsigned long uint64_t;                                                        \n" \
"                                                                                       \n" \
"__kernel void vadd(                                                                    \n" \
"    __global uint64_t* seed_ranges,                                                    \n" \
"    const uint64_t num_seed_ranges,                                                    \n" \
"    __global uint64_t* seed_map_layer_sizes,                                           \n" \
"    __global uint64_t* flat_seed_map_layers,                                           \n" \
"    const uint64_t num_seed_map_layers,                                                \n" \
"    const uint64_t total_map_count,                                                    \n" \
"    __global uint64_t* results                                                         \n" \
")                                                                                      \n" \
"{                                                                                      \n" \
"    uint64_t index = 0;                                                                \n" \
"    uint64_t sml_index = 0;                                                            \n" \
"    uint64_t m_index = 0;                                                              \n" \
"    uint64_t seed_val = 0;                                                             \n" \
"    uint64_t *layer_ptr, *map_ptr;                                                     \n" \
"    uint64_t num_maps;                                                                 \n" \
"    uint64_t seed = 0;                                                                 \n" \
"    uint64_t seed_start = 0;                                                           \n" \
"    uint64_t seed_end = 0;                                                             \n" \
"    uint64_t min_value = ULONG_MAX;                                                    \n" \
"                                                                                       \n" \
"    index = get_global_id(0);                                                          \n" \
"                                                                                       \n" \
"    if(index >= num_seed_ranges) return;                                               \n" \
"                                                                                       \n" \
"    seed_start = seed_ranges[(index * 2)];                                             \n" \
"    seed_end = seed_start + seed_ranges[(index * 2) + 1];                              \n" \
"    layer_ptr = flat_seed_map_layers;                                                  \n" \
"    for (seed = seed_start; seed < seed_end; seed++ )                                  \n" \
"    {                                                                                   \n" \
"        seed_val = seed;                                                                  \n" \
"        for (sml_index = 0; sml_index < num_seed_map_layers; sml_index++)                  \n" \
"        {                                                                                  \n" \
"            num_maps = seed_map_layer_sizes[sml_index];                                    \n" \
"            map_ptr = layer_ptr;                                                           \n" \
"            for (m_index = 0; m_index < num_maps; m_index++)                               \n" \
"            {                                                                              \n" \
"                                                                                           \n" \
"                if ((seed_val >= *(map_ptr)) && (seed_val < *(map_ptr) + *(map_ptr + 2)))  \n" \
"                {                                                                          \n" \
"                    seed_val = seed_val - *(map_ptr) + *(map_ptr + 1);                     \n" \
"                    break;                                                                 \n" \
"                }                                                                          \n" \
"                                                                                           \n" \
"                map_ptr += 3;                                                              \n" \
"            }                                                                              \n" \
"            layer_ptr += (num_maps * 3);                                                   \n" \
"        }                                                                                  \n" \
"        if (seed_val < min_value) min_value = seed_val;                                    \n" \
"    }                                                                                      \n" \
"    results[index] = min_value;                                                            \n" \
"}                                                                                          \n" \
"\n";

//------------------------------------------------------------------------------

uint64_t part_b_opencl(const CONFIG *config)
{
    cl_device_id     device_id;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        ko_vadd;       // compute kernel

    cl_uint numPlatforms;

    // Find number of platforms
    {
        int error;
        if (error = clGetPlatformIDs(0, NULL, &numPlatforms))
        {
            printf("Error getting platform ids");
            return EXIT_FAILURE;
        }
    }
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    {
        int error;
        if(error = clGetPlatformIDs(numPlatforms, Platform, NULL))
        {
            printf("Error getting platform\n");
        }
    }

    // Secure a GPU
    printf("Num platforms %d\n", numPlatforms);
    for (int plat_index = 0; plat_index < numPlatforms; plat_index++)
    {
        int err = clGetDeviceIDs(Platform[plat_index], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (device_id == NULL)
    {
        printf("could not get device\n");
    }

    // Create a compute context
    {
        int error = 0;
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &error);
        if (error)
        {
            printf("Can't create context\n");
            return 1;
        }
    }

    // Create a command queue
    {
        int error = 0;
        // replace with clCreateCommandQueueWithProperties
        commands = clCreateCommandQueue(context, device_id, 0, &error);
        if (error)
        {
            printf("Can't create command queue\n");
            return 1;
        }  
    }

    // Create the compute program from the source buffer
    {
        int error = 0;
        program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource_part_b, NULL, &error);
        if (error)
        {
            printf("Error creating program\n");
            return 1;
        }
    }

    // Build the program
    {
        int error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            size_t len;
            char buffer[2048];

            printf("Error: Failed to build program executable! :%d\n", error);
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf("%s\n", buffer);
            return EXIT_FAILURE;
        }
    }

    // Create the compute kernel from the program
    {
        int error = 0;
        ko_vadd = clCreateKernel(program, "vadd", &error);
        if (error)
        {
            printf("Can't create kernel\n");
            return 1;
        }
    }
    
    SEED_RANGE *seed_ranges;
    uint64_t num_seed_ranges;

    SEED_MAP_LAYER **seed_map_layers;
    uint64_t num_seed_map_layers;

    injest_file_part_b(
        config->input_file_path, 
        &seed_ranges, &num_seed_ranges,
        &seed_map_layers, &num_seed_map_layers
    );

    sort_seed_ranges_by_size(seed_ranges, num_seed_ranges);
    
    SEED_RANGE *new_seed_ranges = split_seed_ranges_by_number(seed_ranges, &num_seed_ranges, 28);
    free(seed_ranges);

    uint64_t total_size;
    uint64_t *flat_seed_map_layer_sizes;
    uint64_t *flat_seed_map_layers = seed_map_layer_flatten_layers(seed_map_layers, num_seed_map_layers, &flat_seed_map_layer_sizes, &total_size);
    seed_map_layers_term(seed_map_layers, num_seed_map_layers);

    cl_mem device_seed_ranges;               // device memory used for the input  a vector
    cl_mem device_seed_map_layers_sizes;     // 
    cl_mem device_flat_seed_map_layers;      // device memory used for the input  b vector
    cl_mem device_results;                   // device memory used for the output c vector

    for (uint64_t ii = 0; ii < num_seed_map_layers; ii++)
    {
        flat_seed_map_layer_sizes[ii] = flat_seed_map_layer_sizes[ii] / 3;
    }

    // Create the seeds device memory
    {
        int error = 0;
        device_seed_ranges  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(SEED_RANGE) * num_seed_ranges, NULL, &error);
        if (error)
        {
            printf("Error creating device seeds variable\n");
        }
    }

    // Create the flat_seed_layers_sizes device memory
    {
        int error = 0;
        device_seed_map_layers_sizes  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(uint64_t) * num_seed_map_layers, NULL, &error);
        if (error)
        {
            printf("Error creating device seeds variable\n");
        }
    }

    // Create the flat_seed_layers device memory
    {
        int error = 0;
        device_flat_seed_map_layers  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(uint64_t) * total_size, NULL, &error);
        if (error)
        {
            printf("Error creating device seeds variable\n");
        }
    }

    // Create the results device memory
    {
        int error = 0;
        device_results  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  sizeof(uint64_t) * num_seed_ranges, NULL, &error);
        if (error)
        {
            printf("Error creating device results variable\n");
        }
    }

    // Write seeds to device memory
    {
        int error = 0;
        error = clEnqueueWriteBuffer(commands, device_seed_ranges, CL_TRUE, 0, sizeof(SEED_RANGE) * num_seed_ranges, new_seed_ranges, 0, NULL, NULL);
        if (error)
        {
            printf("Error copying seeds to device seeds\n");
        }
    }

    // Write layer sizes to device memory
    {
        int error = 0;
        error = clEnqueueWriteBuffer(commands, device_seed_map_layers_sizes, CL_TRUE, 0, sizeof(uint64_t) * num_seed_map_layers, flat_seed_map_layer_sizes, 0, NULL, NULL);
        if (error)
        {
            printf("Error copying seeds to device_seed_map_layers_sizes\n");
        }
    }

    // Write flattened layers to device memory
    {
        int error = 0;
        error = clEnqueueWriteBuffer(commands, device_flat_seed_map_layers, CL_TRUE, 0, sizeof(uint64_t) * total_size, flat_seed_map_layers, 0, NULL, NULL);
        if (error)
        {
            printf("Error copying seeds to device_flat_seed_map_layers\n");
        }
    }

    // // Set the arguments to our compute kernel
    {
        int error = 0;
        error  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &device_seed_ranges);
        error |= clSetKernelArg(ko_vadd, 1, sizeof(cl_ulong), &num_seed_ranges);
        error |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &device_seed_map_layers_sizes);
        error |= clSetKernelArg(ko_vadd, 3, sizeof(cl_mem), &device_flat_seed_map_layers);
        error |= clSetKernelArg(ko_vadd, 4, sizeof(cl_ulong), &num_seed_map_layers);
        error |= clSetKernelArg(ko_vadd, 5, sizeof(cl_ulong), &total_size);
        error |= clSetKernelArg(ko_vadd, 6, sizeof(cl_mem), &device_results);

        if (error)
        {
            printf("Error setting variables to kernel\n");
            return 1;
        }
    }

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    {
        int error = 0;
        uint64_t global = num_seed_ranges;
        error = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
        if (error)
        {
            printf("Error enquing kernel\n");
            return 1;
        }

        // Wait for the commands to complete before stopping the timer
        error = clFinish(commands);
        if (error)
        {
            printf("Error finishing kernel\n");
            return 1;
        }
    }

    uint64_t *results = (uint64_t *)calloc(num_seed_ranges, sizeof(uint64_t));
    // Read back the results from the compute device
    {
        int error = clEnqueueReadBuffer( commands, device_results, CL_TRUE, 0, sizeof(uint64_t) * num_seed_ranges, results, 0, NULL, NULL );  
        if (error != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! : %d\n", error);
            exit(1);
        }
    }   

    uint64_t min_value = UINT64_MAX;
    for(uint64_t results_index = 0; results_index < num_seed_ranges; results_index++)
    {
        if (results[results_index] < min_value) min_value = results[results_index];
    }

    free(results);
    free(new_seed_ranges);
    free(flat_seed_map_layer_sizes);
    free(flat_seed_map_layers);
    
    return min_value;
}
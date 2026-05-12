// 05_compute_speed.cpp
#include "common.hpp"

int main()
{
    // Load velocity data from binary file
    

    // Each particle has 3 velocity components (vx, vy, vz)
    // So the number of particles is total floats divided by 3
    


    // Setup OpenCL context, device, and command queue with profiling enabled
    
    


    // Create buffers for velocity input and speed output
    
    

    // Load and build the OpenCL kernel
    
    


    // Set kernel arguments and execute.
    
    


    // Test different local work sizes and profile execution time
    // Note: The "AUTO" case lets OpenCL decide the local size, which may not always be optimal.
    // Try sizes: 0 (AUTO), 32, 64, 128
    
    


    // For each local size, enqueue the kernel and measure execution time using OpenCL events.
        // - If local size is 0, use AUTO (let OpenCL decide). Otherwise, set the specified local size and calculate the global size accordingly.
        // - Enqueue the kernel and wait for it to finish, then read profiling info from the event.
        // - Get profiling info (start and end time) from the event to calculate elapsed time in milliseconds.
        // - Print the profiling results for this local size configuration.
        

    // Read back the computed speeds to verify correctness (optional).
    
    

    // Print the first few computed speeds to verify correctness.
    
    
    

    return 0;
}

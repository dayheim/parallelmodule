// ============================================================
// This kernel should perform the operation:
//
//     C = A + B
//
// ============================================================
kernel void vector_add(global const float* A, global const float* B, global float* C) {
    
    int id = get_global_id(0);

    // TODO #1: Print global ID using printf
    // ---------------------------------------------------------
    // get_global_id(0) returns the index of this work item in
    // the global NDRange along the X dimension.
    //
    // printf() is supported in OpenCL 1.2+ on many GPUs/CPUs
    // (but not all - some embedded/OpenCL-1.1 devices may not
    // provide printf support).
    //
    // This will print once per work item and is useful for
    // debugging execution order or ensuring correct dispatch.
    // ---------------------------------------------------------

    
    


    // ---------------------------------------------------------
    // TODO #2:
    // Print work-group info.
    // `get_local_size(0)` returns number of work-items
    // per group in dimension 0.
    //
    // Only print once using id == 0 to avoid huge output.
    // ---------------------------------------------------------
    
    




    // ---------------------------------------------------------
    // TODO #3:
    // Print total work-items (once) +
    // Print work-group ID for each work-item
    // ---------------------------------------------------------

    
    
    


    // Perform vector addition
    C[id] = A[id] + B[id];
}

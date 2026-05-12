// 02_gravity.cpp  (Apply Gravity)
#include "common.hpp"

int main()
{
    /* --------------------------------------------------------
    TODO #1: Choose dataset: 
    "small_data.bin"  (1,000 particles) for debugging
    "large_data.bin"  (100,000 particles) for profiling
    --------------------------------------------------------*/
    std::string filename = "dataset/small_data.bin";

    std::vector<float> velocities;
    if (!loadSimulationData(filename, velocities)) {
        return -1;
    }

    /*
    Quick sanity check on loaded data
    - Remember: data is in Structure of Arrays format:
    [vx0, vy0, vz0, vx1, vy1, vz1, ..., vxN-1, vyN-1, vzN-1]
    So numParticles = total floats / 3 (for vx, vy, vz)
    This is important for kernel bounds checks and NDRange sizes.
    */
    int numParticles = static_cast<int>(velocities.size() / 3);
    std::cout << "Loaded " << numParticles << " particles from "
              << filename << "\n";
    printFirstParticles(velocities);

    // setup OpenCL context, device, and command queue
    // setupOpenCL function - see common.hpp for details and device selection logic
    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    if (!setupOpenCL(context, device, queue, CL_DEVICE_TYPE_GPU)) {
        return -1;
    }

    // Create OpenCL buffers for input velocities and output new velocities.
    cl::Buffer d_vel(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * velocities.size(), velocities.data());
    cl::Buffer d_newVel(context, CL_MEM_WRITE_ONLY, sizeof(float) * velocities.size());

    // Load and build the OpenCL kernel from file.
    std::string kernelSource = loadKernelSource("kernels/gravity.cl");
    if (kernelSource.empty()) return -1;

    cl::Program program(context, kernelSource);
    cl_int buildErr = program.build({device});
    if (buildErr != CL_SUCCESS) {
        std::cerr << "Build error: " << buildErr << "\n";
        std::cerr << "Build log:\n"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        return -1;
    }

    // --------------------------------------------------------
    // TODO #3: Set gravity (g) and time step (dt).
    // Example: g = (0, -9.81, 0)--> g.x=0.0f; g.y=-9.81f; g.z=0.0f; and dt = 0.01
    // You can experiment with different values.
    // --------------------------------------------------------
    cl_float3 g;
    float dt;
    // initialize gravity vector and time step (dt) here
    
    


    cl::Kernel kernel(program, "apply_gravity");
    // Set kernel arguments (match the order in gravity.cl)
    // - Remember to pass numParticles for bounds check in kernel!
    // use kernel.setArg() to set each argument (see OpenCL docs or examples)
    
    



    /* ========================================================
    PROFILING SECTION (V3: padded NDRange)
    We test: AUTO local size (OpenCL chooses, i.e. 0) and everal manual local sizes (e.g. 32, 64, 128)
    For manual sizes we: 
    1. Round global size UP to next multiple of local size.
    2. Use bounds check (id >= numParticles) in kernel.
    ========================================================*/
    std::vector<size_t> localSizes = {
        0,   // 0 => AUTO
        32,
        64,
        128
    };

    for (size_t ls : localSizes) {
        std::string label;
        cl::NDRange global;
        cl::NDRange local;
        cl::Event event;

        if (ls == 0) {
            // AUTO: let OpenCL decide local size
            label = "AUTO";
            global = cl::NDRange(numParticles);
            local  = cl::NullRange;
        } else {
            label = "localSize=" + std::to_string(ls);
            size_t padded = roundUpToMultiple(numParticles, ls);
            global = cl::NDRange(padded);
            local  = cl::NDRange(ls);
        }

        queue.enqueueNDRangeKernel(kernel,
                                   cl::NullRange,
                                   global,
                                   local,
                                   nullptr,
                                   &event);
        queue.finish();

        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double elapsedMs = (end - start) * 1.0e-6;

        std::cout << "[Profiling] " << label
                  << " : " << elapsedMs << " ms\n";
    }

    // Read back one of the results for correctness check
    std::vector<float> newVelocities(velocities.size());
    queue.enqueueReadBuffer(d_newVel, CL_TRUE, 0,
                            sizeof(float) * newVelocities.size(),
                            newVelocities.data());

    std::cout << "Updated velocities (first 5):\n";
    printFirstParticles(newVelocities);

    /* 
    TODO #4 (REPORT):Record the runtimes for AUTO and manual local sizes. 
    Identify the best configuration and explain why.
    */


    

    return 0;
}

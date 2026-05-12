// UpdateLifetimes.cpp
#include "common.hpp"
#include <random>

int main()
{
    // load particle velocities (not used in this example, but typically part of the simulation data)
    std::string filename = "dataset/small_data.bin"; 
    std::vector<float> velocities;
    if (!loadSimulationData(filename, velocities)) {
        return -1;
    }

    int numParticles = static_cast<int>(velocities.size() / 3);
    std::cout << "Loaded " << numParticles << " particles from " << filename << "\n";

    // --------------------------------------------------------
    // Initialize lifetimes.
    // Example: random values in [0, 5].
    // --------------------------------------------------------
    std::vector<float> lifetimes(numParticles);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 5.0f);
    for (int i = 0; i < numParticles; ++i) {
        lifetimes[i] = dist(rng);
    }

    float dt = 0.1f; // time step

    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    if (!setupOpenCL(context, device, queue, CL_DEVICE_TYPE_GPU)) {
        return -1;
    }

    cl::Buffer d_lifetimes(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * numParticles, lifetimes.data());
    cl::Buffer d_alive(context, CL_MEM_WRITE_ONLY, sizeof(char) * numParticles);

    std::string kernelSource = loadKernelSource("kernels/lifetime.cl");
    if (kernelSource.empty()) return -1;

    cl::Program program(context, kernelSource);
    cl_int buildErr = program.build({device});
    if (buildErr != CL_SUCCESS) {
        std::cerr << "Build error: " << buildErr << "\n";
        std::cerr << "Build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        return -1;
    }

    cl::Kernel kernel(program, "update_lifetimes");
    kernel.setArg(0, d_lifetimes);
    kernel.setArg(1, d_alive);
    kernel.setArg(2, dt);
    kernel.setArg(3, numParticles);

    std::vector<size_t> localSizes = {
        0, 32, 64, 128
    };

    for (size_t ls : localSizes) {
        std::string label;
        cl::NDRange global;
        cl::NDRange local;
        cl::Event event;

        if (ls == 0) {
            label = "AUTO";
            global = cl::NDRange(numParticles);
            local  = cl::NullRange;
        } else {
            label = "localSize=" + std::to_string(ls);
            size_t padded = roundUpToMultiple(numParticles, ls);
            global = cl::NDRange(padded);
            local  = cl::NDRange(ls);
        }

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &event);
        queue.finish();

        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double elapsedMs = (end - start) * 1.0e-6;

        std::cout << "[Profiling] " << label << " : " << elapsedMs << " ms\n";
    }

    std::vector<char> alive(numParticles);
    queue.enqueueReadBuffer(d_alive, CL_TRUE, 0, sizeof(char) * numParticles, alive.data());

    int aliveCount = 0;
    for (int i = 0; i < numParticles; ++i)
        if (alive[i] == 1) ++aliveCount;

    std::cout << "Alive particles after dt = " << dt << " : " << aliveCount << " / " << numParticles << "\n";

    return 0;
}

#include "common.hpp"
#include <cmath>

int main()
{
    std::string filename = "dataset/small_data.bin"; // switch to large_data.bin for profiling
    std::vector<float> velocities;
    if (!loadSimulationData(filename, velocities)) {
        return -1;
    }

    int numParticles = static_cast<int>(velocities.size() / 3);
    std::cout << "Loaded " << numParticles << " particles from " << filename << "\n";

    /*--------------------------------------------------------
    Derive force and mass arrays. 
    F[i] proportional to velocity
    m[i] slightly varying positive values
    --------------------------------------------------------*/
    std::vector<cl_float3> forces(numParticles);
    std::vector<float> masses(numParticles);

    for (int i = 0; i < numParticles; ++i) {
        float vx = velocities[3*i + 0];
        float vy = velocities[3*i + 1];
        float vz = velocities[3*i + 2];

        forces[i].x = 0.1f * vx;
        forces[i].y = 0.1f * vy;
        forces[i].z = 0.1f * vz;

        // Mass formula: m_i = base + step * (i % cycle)
        // Ensures non-zero mass and small variation.
        masses[i] = 1.0f + 0.01f * (i % 100);
    }

    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    if (!setupOpenCL(context, device, queue, CL_DEVICE_TYPE_GPU)) {
        return -1;
    }

    cl::Buffer d_force(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float3) * numParticles, forces.data());
    cl::Buffer d_mass(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * numParticles, masses.data());
    cl::Buffer d_accel(context, CL_MEM_WRITE_ONLY, sizeof(cl_float3) * numParticles);

    std::string kernelSource = loadKernelSource("kernels/acceleration.cl");
    if (kernelSource.empty()) return -1;

    cl::Program program(context, kernelSource);
    cl_int buildErr = program.build({device});
    if (buildErr != CL_SUCCESS) {
        std::cerr << "Build error: " << buildErr << "\n";
        std::cerr << "Build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        return -1;
    }

    cl::Kernel kernel(program, "compute_accel");
    kernel.setArg(0, d_force);
    kernel.setArg(1, d_mass);
    kernel.setArg(2, d_accel);
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

    std::vector<cl_float3> accel(numParticles);
    queue.enqueueReadBuffer(d_accel, CL_TRUE, 0, sizeof(cl_float3) * numParticles, accel.data());

    std::cout << "Sample accelerations (first 5):\n";
    for (int i = 0; i < std::min(numParticles, 5); ++i) {
        std::cout << "  a[" << i << "] = (" << accel[i].x << ", " << accel[i].y << ", " << accel[i].z << ")\n";
    }

    return 0;
}

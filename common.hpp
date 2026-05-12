#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 300   
// #define CL_HPP_MINIMUM_OPENCL_VERSION 120  // but still support OpenCL 1.2 (NVIDIA GPUs)
#define CL_HPP_ENABLE_EXCEPTIONS           // wrapper will throw C++ exceptions on errors

#include <CL/opencl.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

// (optional but encouraged for safety/error reporting)
#include <stdexcept>


// =======================================================================
// FUNCTION: loadSimulationData()
// PURPOSE:  Loads 1D array of floats representing particle attributes.
//
// IMPORTANT for you:
// - Particles are stored in *Structure of Arrays* format:
//
//     [vx0, vy0, vz0, vx1, vy1, vz1, ..., vxN-1, vyN-1, vzN-1]
//
//   This matches what GPUs like: contiguous memory for each attribute
//   → better memory coalescing → faster GPU performance.
//
// - The binary data files provided are large to simulate real workloads.
// =======================================================================
inline bool loadSimulationData(const std::string& filename,
                               std::vector<float>& dataOut)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << "\n";
        return false;
    }

    // Determine file size (in bytes), then convert to number of floats.
    file.seekg(0, std::ios::end);
    std::streamoff fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize <= 0) {
        std::cerr << "Error: File is empty or invalid: " << filename << "\n";
        return false;
    }

    size_t numFloats = static_cast<size_t>(fileSize) / sizeof(float);
    dataOut.resize(numFloats);

    // Read all bytes directly into float array memory
    file.read(reinterpret_cast<char*>(dataOut.data()), fileSize);
    file.close();

    return true;
}


// =======================================================================
// FUNCTION: loadKernelSource()
// PURPOSE:  Reads OpenCL kernel (.cl) into a string for program.build().
//
// - Kernels are written in OpenCL C, not C++
// - Host passes source be compiled ON THE GPU DRIVER at runtime!
// =======================================================================
inline std::string loadKernelSource(const std::string& filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open kernel file: " << filePath << "\n";
        return "";
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}


// =======================================================================
// FUNCTION: roundUpToMultiple()
// PURPOSE:  Ensures padded global size divisible by local size.
//
// Why is this needed?
// - Older OpenCL 1.2 GPUs (esp. NVIDIA) require:
//
//       globalSize % localSize == 0
//
// - Newer OpenCL 2.x can handle mismatch if kernel bounds-checks index
//
// We ALWAYS do padding + bounds check so it works on every device.
// =======================================================================
inline size_t roundUpToMultiple(size_t value, size_t divisor)
{
    if (divisor == 0) return value;  // AUTO mode → no change
    size_t remainder = value % divisor;
    if (remainder == 0) return value;
    return value + (divisor - remainder);
}


// =======================================================================
// FUNCTION: setupOpenCL()
// PURPOSE:
//   - Print available devices (for your lab report)
//   - Allow selection CPU vs GPU via desiredType flag
//   - Enable profiling for performance measurements
//
// You SHOULD:
//   - Capture printed device info in lab results!
//   - Experiment running on CPU vs GPU
//   - Observe huge performance difference on large data!
//
// NOTE:
// - Using CL_QUEUE_PROFILING_ENABLE is essential for timing kernels
//   (required in Task4U performance analysis)
// =======================================================================
inline bool setupOpenCL(cl::Context& context,
                        cl::Device& device,
                        cl::CommandQueue& queue,
                        cl_device_type desiredType = CL_DEVICE_TYPE_GPU)
{
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found.\n";
            return false;
        }

        std::cout << "\n=== Available OpenCL Platforms and Devices ===\n";

        cl::Platform selectedPlatform;
        cl::Device selectedDevice;
        bool deviceFound = false;

        for (size_t p = 0; p < platforms.size(); ++p) {
            auto& platform = platforms[p];
            std::cout << "Platform " << p << ": "
                      << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            for (size_t d = 0; d < devices.size(); ++d) {
                auto& dev = devices[d];

                cl_device_type dtype = dev.getInfo<CL_DEVICE_TYPE>();

                // Simple classification
                std::string typeStr =
                    (dtype & CL_DEVICE_TYPE_GPU) ? "GPU" :
                    (dtype & CL_DEVICE_TYPE_CPU) ? "CPU" :
                    "Other (accelerator)";

                std::cout << "   Device " << d << ": "
                          << dev.getInfo<CL_DEVICE_NAME>()
                          << "  [" << typeStr << "]\n";

                // ⬇ AUTO-SELECT FIRST DESIRED TYPE (Default = GPU)
                if (!deviceFound && (dtype & desiredType)) {
                    selectedPlatform = platform;
                    selectedDevice   = dev;
                    deviceFound      = true;
                }
            }
        }

        if (!deviceFound) {
            std::cerr << "No requested device type found → using first available.\n";
            selectedPlatform = platforms[0];
            std::vector<cl::Device> devs;
            selectedPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devs);
            if (devs.empty()) {
                std::cerr << "No OpenCL devices found.\n";
                return false;
            }
            selectedDevice = devs[0];
        }

        std::cout << "\n>> Using device: "
                  << selectedDevice.getInfo<CL_DEVICE_NAME>()
                  << "\n";

        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(selectedPlatform)(),
            0
        };

        context = cl::Context(selectedDevice, props);

        // Profiling enabled: This is REQUIRED for kernel timing metrics
        queue = cl::CommandQueue(
            context,
            selectedDevice,
            CL_QUEUE_PROFILING_ENABLE
        );

        device = selectedDevice;
    }
    catch (...) {
        std::cerr << "Unknown error during OpenCL setup.\n";
        return false;
    }

    return true;
}


// =======================================================================
// FUNCTION: printFirstParticles()
// PURPOSE:  Quick verification of data correctness.
//
// Why?
// - GPU debugging is difficult.
// - Visual sanity checks help detect incorrect kernel behavior quickly.
// =======================================================================
inline void printFirstParticles(const std::vector<float>& data,
                                int numParticlesToPrint = 5)
{
    int maxParticles = static_cast<int>(data.size() / 3);
    numParticlesToPrint = std::min(numParticlesToPrint, maxParticles);

    std::cout << "First " << numParticlesToPrint << " particles:\n";
    for (int i = 0; i < numParticlesToPrint; ++i) {
        std::cout << "  p[" << i << "] = ("
                  << data[3*i + 0] << ", "
                  << data[3*i + 1] << ", "
                  << data[3*i + 2] << ")\n";
    }
}

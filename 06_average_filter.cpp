// 06_average_filter

#include "common.hpp"

int main()
{
    // --------------------------------------------------------
    // Choose the signal dataset (binary floats)
    // --------------------------------------------------------
    std::string filename = "dataset/small_signal.bin";

    // --------------------------------------------------------
    // Load input signal from file into host vector.
    // --------------------------------------------------------
    std::vector<float> input;
    if (!loadSimulationData(filename, input)) {
        return -1;
    }

    if (input.empty()) {
        std::cerr << "Input is empty.\n";
        return -1;
    }

    const int N = static_cast<int>(input.size());

    // --------------------------------------------------------
    // Print first few samples for verification
    // --------------------------------------------------------
    std::cout << "Loaded " << input.size() << " samples from " << filename << "\n";
    std::cout << "First 10 input samples:\n";
    for (size_t i = 0; i < std::min(input.size(), static_cast<size_t>(10)); ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << "\n";

    // --------------------------------------------------------
    // Filter parameter (must be positive and odd)
    // --------------------------------------------------------
    int k = 5;
    if (k <= 0 || (k % 2) == 0) {
        std::cerr << "k must be positive and odd. Got k=" << k << "\n";
        return -1;
    }

    // --------------------------------------------------------
    // Allocate host output
    // --------------------------------------------------------
    std::vector<float> output(input.size(), 0.0f);

    // --------------------------------------------------------
    // OpenCL setup
    // --------------------------------------------------------
    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;

    if (!setupOpenCL(context, device, queue, CL_DEVICE_TYPE_GPU)) {
        return -1;
    }

    // Ensure profiling is enabled (needed for event.getProfilingInfo)
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // --------------------------------------------------------
    // Buffers
    // --------------------------------------------------------
    cl::Buffer d_input(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * input.size(),
        input.data()
    );

    // Output buffer: no need to COPY_HOST_PTR (we'll write into it on device)
    cl::Buffer d_out(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(float) * output.size()
    );

    // --------------------------------------------------------
    // Program build
    // --------------------------------------------------------
    std::string kernelSource = loadKernelSource("kernels/avg_filter.cl");
    if (kernelSource.empty()) return -1;

    cl::Program program(context, kernelSource);
    cl_int buildErr = program.build({ device });
    if (buildErr != CL_SUCCESS) {
        std::cerr << "Build error: " << buildErr << "\n";
        std::cerr << "Build log:\n"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        return -1;
    }

    // --------------------------------------------------------
    // Kernel args:
    // __kernel void avg_filter(__global const float* input,
    //                          __global float* output,
    //                          int N,
    //                          int k)
    // --------------------------------------------------------
    cl::Kernel kernel(program, "avg_filter_valid");
    kernel.setArg(0, d_input);
    kernel.setArg(1, d_out);
    kernel.setArg(2, N);
    kernel.setArg(3, k);

    // --------------------------------------------------------
    // Launch (AUTO local size)
    // global = N work-items
    // --------------------------------------------------------
    cl::Event event;
    queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(static_cast<size_t>(N)),
        cl::NullRange,
        nullptr,
        &event
    );
    queue.finish();

    // Profiling
    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double elapsedMs = (end - start) * 1.0e-6;
    std::cout << "[Profiling] AUTO : " << elapsedMs << " ms\n";

    // --------------------------------------------------------
    // Read back
    // --------------------------------------------------------
    queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(float) * output.size(), output.data());

    std::cout << "Output size = " << output.size() << "\n";
    std::cout << "Output (first 10 average filtered values):\n";
    for (int i = 0; i < std::min(static_cast<int>(output.size()), 10); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << "\n";

    return 0;
}

// 06_average_filter_2D_planarRGB
// g++ 07_average_filterND.cpp -o  07_average_filterND.exe -I"C:\OpenCL-SDK\include" -lOpenCL -lgdi32 -luser32 -lkernel32 -lpng -lz

// -lgdi32 -luser32 -lkernel32 are needed for CImg display (windows). 
// -lOpenCL is needed for OpenCL.
// -lpng -lz are needed for CImg PNG support (if you have PNG images).

#include "common.hpp"
#include "CImg.h"

using namespace cimg_library;

int main()
{
    // --------------------------------------------------------
    // Load image (PPM) using CImg
    // --------------------------------------------------------
    const char* filename = "dataset/opencv_small.png";
    CImg<unsigned char> img(filename);

    int width    = img.width();
    int height   = img.height();
    int channels = img.spectrum(); // should be 3 for RGB

    if (channels != 3) {
        std::cerr << "Expected RGB image (3 channels). Got channels=" << channels << "\n";
        return -1;
    }

    // --------------------------------------------------------
    // Filter parameter (must be positive odd)
    // --------------------------------------------------------
    int k = 5;
    if (k <= 0 || (k % 2) == 0) {
        std::cerr << "k must be positive and odd. Got k=" << k << "\n";
        return -1;
    }
    int r = k / 2;

    // If k is too large, there is no valid interior
    if (width <= 2 * r || height <= 2 * r) {
        std::cerr << "k too large for this image. width=" << width
                  << " height=" << height << " k=" << k << "\n";
        return -1;
    }

    const size_t image_size = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t total_size = image_size * static_cast<size_t>(channels);

    // --------------------------------------------------------
    // Convert CImg (x,y,c) -> planar buffer
    // planar index: x + y*width + c*(width*height)
    // --------------------------------------------------------
    std::vector<unsigned char> input(total_size);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                input[static_cast<size_t>(x) + static_cast<size_t>(y) * width + static_cast<size_t>(c) * image_size]
                    = img(x, y, 0, c);
            }
        }
    }

    // Output: initialize deterministically.
    // For VALID-ONLY kernel that "returns" at borders, this avoids garbage edges.
    // Choose either:
    //  - output = input  (keep original borders), or
    //  - output = 0      (black borders)
    std::vector<unsigned char> output = input; // keep borders unchanged

    // --------------------------------------------------------
    // OpenCL setup
    // --------------------------------------------------------
    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;

    if (!setupOpenCL(context, device, queue, CL_DEVICE_TYPE_GPU)) {
        return -1;
    }
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // --------------------------------------------------------
    // Buffers (uchar planar)
    // --------------------------------------------------------
    cl::Buffer d_input(context,
                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(unsigned char) * input.size(),
                       input.data());

    // For VALID-ONLY kernel that may not write borders, seed device output with host output
    cl::Buffer d_out(context,
                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(unsigned char) * output.size(),
                     output.data());

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
    // __kernel void avg_filterND(__global const uchar* A,
    //                            __global uchar* B,
    //                            int width, int height, int channels, int k)
    // --------------------------------------------------------
    cl::Kernel kernel(program, "avg_filterND");
    kernel.setArg(0, d_input);
    kernel.setArg(1, d_out);
    kernel.setArg(2, width);
    kernel.setArg(3, height);
    kernel.setArg(4, channels);
    kernel.setArg(5, k);

    // --------------------------------------------------------
    // Launch configuration for VALID-ONLY kernel
    //
    // Option 1: launch full image and let kernel skip borders.
    // This works because the kernel has:
    //   if (x >= width || y >= height || c >= channels) return;
    //   if (x < r || x >= width-r || y < r || y >= height-r) return;
    // and because d_out is pre-seeded with original border values.
    // --------------------------------------------------------
    cl::NDRange global(static_cast<size_t>(width),
                       static_cast<size_t>(height),
                       static_cast<size_t>(channels));

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    queue.finish();

    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double elapsedMs = (end - start) * 1.0e-6;
    std::cout << "[Profiling] avg_filterND (valid-only) : " << elapsedMs << " ms\n";

    // --------------------------------------------------------
    // Read back
    // --------------------------------------------------------
    queue.enqueueReadBuffer(d_out, CL_TRUE, 0,
                            sizeof(unsigned char) * output.size(),
                            output.data());

    // --------------------------------------------------------
    // Convert planar output -> CImg for display
    // --------------------------------------------------------
    CImg<unsigned char> out_img(width, height, 1, channels);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                out_img(x, y, 0, c) =
                    output[static_cast<size_t>(x) + static_cast<size_t>(y) * width + static_cast<size_t>(c) * image_size];
            }
        }
    }

    CImgDisplay disp_in(img, "Input");
    CImgDisplay disp_out(out_img, "Average Filter Output (valid-only interior)");

    while (!disp_in.is_closed() && !disp_out.is_closed()) {
        disp_in.wait(10);
        disp_out.wait(10);
    }

    return 0;
}

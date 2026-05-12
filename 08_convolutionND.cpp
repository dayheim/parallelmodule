// 08_convolution_2D_planarRGB_valid
// g++ 08_convolutionND.cpp -o 08_convolutionND.exe -I"C:\OpenCL-SDK\include" -lOpenCL -lgdi32 -luser32 -lkernel32 -lpng -lz
//
// -lgdi32 -luser32 -lkernel32 are needed for CImg display (Windows).
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
    // Convolution parameter (k must be positive odd)
    // --------------------------------------------------------
    int k = 21;
    if (k <= 0 || (k % 2) == 0) {
        std::cerr << "k must be positive and odd. Got k=" << k << "\n";
        return -1;
    }
    int r = k / 2;

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

    // VALID-ONLY kernel returns at borders, so seed output to keep borders unchanged
    std::vector<unsigned char> output = input;

    // --------------------------------------------------------
    // TODO: Build a k×k mask (example: normalized box blur)
    // Replace this with your own mask values if needed.
    // --------------------------------------------------------
    // std::vector<float> mask(static_cast<size_t>(k) * static_cast<size_t>(k), 1.0f / (float)(k * k));

    if ((int)mask.size() != k * k) {
        std::cerr << "Mask size mismatch: mask.size()=" << mask.size() << " but k*k=" << (k*k) << "\n";
        return -1;
    }

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
    // Buffers
    // --------------------------------------------------------
    cl::Buffer d_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * input.size(), input.data());
    cl::Buffer d_out(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * output.size(), output.data());
    // cl::Buffer d_mask(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * mask.size(), mask.data());

    // --------------------------------------------------------
    // Program build
    // --------------------------------------------------------
    std::string kernelSource = loadKernelSource("kernels/convolution_filter.cl"); // contains convolutionND too
    if (kernelSource.empty()) return -1;

    cl::Program program(context, kernelSource);
    cl_int buildErr = program.build({ device }); 
    if (buildErr != CL_SUCCESS) {
        std::cerr << "Build error: " << buildErr << "\n";
        std::cerr << "Build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        return -1;
    }

    // --------------------------------------------------------
    // Kernel args for:
    // __kernel void convolutionND(A, B, mask, width, height, channels, k)
    // --------------------------------------------------------
    cl::Kernel kernel(program, "convolutionND");
    kernel.setArg(0, d_input);
    kernel.setArg(1, d_out);
    // kernel.setArg(2, d_mask);
    kernel.setArg(3, width);
    kernel.setArg(4, height);
    kernel.setArg(5, channels);
    kernel.setArg(6, k);

    // --------------------------------------------------------
    // Launch full image; kernel itself is VALID-ONLY and skips borders.
    // Output buffer is pre-seeded so borders remain unchanged.
    // --------------------------------------------------------
    cl::NDRange global(static_cast<size_t>(width), static_cast<size_t>(height), static_cast<size_t>(channels));

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    queue.finish();

    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double elapsedMs = (end - start) * 1.0e-6;
    std::cout << "[Profiling] convolutionND (valid-only) : " << elapsedMs << " ms\n";

    // --------------------------------------------------------
    // Read back
    // --------------------------------------------------------
    queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(unsigned char) * output.size(), output.data());

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
    CImgDisplay disp_out(out_img, "Convolution Output (valid-only interior)");

    while (!disp_in.is_closed() && !disp_out.is_closed()) {
        disp_in.wait(10);
        disp_out.wait(10);
    }

    return 0;
}

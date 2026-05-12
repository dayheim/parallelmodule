
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

std::string loadKernelSource(const std::string& filePath) {
    std::ifstream file(filePath);
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

int main() {
    const int N = 10;
    std::vector<float> A(N), B(N), C(N);
    for(int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2.0f;
    }

    std::string kernelSource = loadKernelSource("kernels/vector_add.cl");

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for(size_t p = 0; p < platforms.size(); p++) {
        auto& platform = platforms[p];

        std::cout << "\n=== Platform " << p << ": "
                  << platform.getInfo<CL_PLATFORM_NAME>() << " ===\n";

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for(size_t d = 0; d < devices.size(); d++) {
            auto& device = devices[d];

            std::cout << "-> Running on device " << d << ": "
                      << device.getInfo<CL_DEVICE_NAME>() << "\n";

            try {
                cl::Context context(device);
                cl::CommandQueue queue(context, device);

                cl::Buffer d_A(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                               sizeof(float)*N, A.data());
                cl::Buffer d_B(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                               sizeof(float)*N, B.data());
                cl::Buffer d_C(context, CL_MEM_WRITE_ONLY,
                               sizeof(float)*N);

                cl::Program program(context, kernelSource);
                program.build({device});

                cl::Kernel kernel(program, "vector_add");
                kernel.setArg(0, d_A);
                kernel.setArg(1, d_B);
                kernel.setArg(2, d_C);

                // =====================================================================
                // TODO #5: Query recommended minimum work-group size multiple
                // =====================================================================
                


                // =====================================================================
                // TODO #6: Query maximum work-group size allowed for this kernel
                // =====================================================================
                
                

                // =====================================================================
                // TODO #7: Choose a valid local size using preferred alignment (simple strategy)
                // =====================================================================



                // =====================================================================
                // TODO #8: Modify to launch with explicitly chosen local size
                // =====================================================================

                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
                queue.finish();

                queue.enqueueReadBuffer(d_C, CL_TRUE, 0,
                                        sizeof(float)*N, C.data());

                std::cout << "   First 10 results: ";
                for(int i = 0; i < 10; i++)
                    std::cout << C[i] << " ";
                std::cout << "\n";
            }
            catch(cl::Error& err) {
                std::cerr << "   [ERROR] " << err.what()
                          << " (" << err.err() << ")\n";
            }
        }
    }

    return 0;
}

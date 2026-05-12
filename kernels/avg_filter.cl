__kernel void avg_filter_valid(__global const float* input,
                         __global float* output,
                         int N,
                         int k)
{
    int gid = get_global_id(0);
    int r = k / 2;

    // Only process valid positions where the full window exists
    // it does not compute output values at the boundaries.
    // This is usually called “valid” convolution behavior:
    if (gid < r || gid >= N - r) return;

    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        sum += input[gid + i];
    }

    output[gid] = sum / (float)k;
}

__kernel void avg_filter_zeropad(__global const float* input,
                                 __global float* output,
                                 int N,
                                 int k)
{
    // TODO: Complete these codes
    
}

__kernel void avg_filter_clamp(__global const float* input,
                               __global float* output,
                               int N,
                               int k)
{
    // TODO: Complete these codes


}


// 2D k×k averaging filter (planar RGB) — VALID-ONLY
__kernel void avg_filterND(__global const uchar* A,
                           __global uchar* B,
                           int width,
                           int height,
                           int channels,
                           int k)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2);

    // Safety guard for launch size
    if (x >= width || y >= height || c >= channels) return;

    int r = k / 2;

    // VALID-ONLY: require full k×k window to fit
    if (x < r || x >= (width  - r) ||
        y < r || y >= (height - r)) {
        return;   // or explicitly set output to 0 if required
    }

    int image_size = width * height;
    int id = x + y * width + c * image_size;

    uint sum = 0;

    for (int j = y - r; j <= y + r; ++j)
        for (int i = x - r; i <= x + r; ++i)
            sum += A[i + j * width + c * image_size];

    B[id] = (uchar)(sum / (uint)(k * k));
}

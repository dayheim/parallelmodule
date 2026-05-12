// 2D k×k averaging filter (planar RGB) — NO boundary handling
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

    // Still keep this safety check unless host *guarantees* exact bounds.
    if (x >= width || y >= height || c >= channels) return;

    int image_size = width * height;
    int id = x + y * width + c * image_size;

    int r = k / 2;
    uint sum = 0;

    for (int j = y - r; j <= y + r; ++j)
        for (int i = x - r; i <= x + r; ++i)
            sum += A[i + j * width + c * image_size];

    B[id] = (uchar)(sum / (uint)(k * k));
}

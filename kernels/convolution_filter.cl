// 2D k×k convolution filter (planar RGB) — VALID-ONLY
__kernel void convolutionND(__global const uchar* A,
                            __global uchar* B,
                            __constant float* mask,
                            int width,
                            int height,
                            int channels,
                            int k)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int c = get_global_id(2);

    if (x >= width || y >= height || c >= channels) return;

    const int r = k / 2;

    // VALID-ONLY region
    if (x < r || x >= (width - r) || y < r || y >= (height - r)) {
        return; // host pre-seeded borders
    }

    const size_t image_size = (size_t)width * (size_t)height;
    const size_t id = (size_t)x + (size_t)y * (size_t)width + (size_t)c * image_size;

    float sum = 0.0f;

    for (int j = -r; j <= r; ++j) {
        const int yy = y + j;
        const int mask_y = j + r;

        for (int i = -r; i <= r; ++i) {
            const int xx = x + i;
            const int mask_x = i + r;

            const size_t a_idx = (size_t)xx + (size_t)yy * (size_t)width + (size_t)c * image_size;
            const int m_idx = mask_x + mask_y * k;

            sum += (float)A[a_idx] * mask[m_idx];
        }
    }

    // Saturating conversion avoids wraparound on negatives and >255 values
    B[id] = convert_uchar_sat_rte(sum);
    //B[id] = (uchar)sum;
}

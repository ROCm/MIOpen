#include <hip/hip_runtime.h>
#include "MIOpenCol2Im3d.cpp"

#include <iostream>
#include <algorithm>
#include <cmath>

#define CHECK(error)                                      \
    if (error != hipSuccess) {                            \
        std::cerr << "HIP error: " << hipGetErrorString(error) << std::endl; \
        exit(1);                                           \
    }

void Col2Im3d_host(_FLOAT* col,
                  const int col_d,
                  const int col_h,
                  const int col_w,
                  const int wei_d,
                  const int wei_h,
                  const int wei_w,
                  const int pad_d,
                  const int pad_h,
                  const int pad_w,
                  const int stride_d,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_d,
                  const int dilation_h,
                  const int dilation_w,
                  const int depth,
                  const int height,
                  const int width,
                  _FLOAT* im,
                  const unsigned long im_offset) {

    // Initialize the output image array
    for (int i = 0; i < depth * height * width; ++i) {
        im[i + im_offset] = 0;
    }

    for (int gid = 0; gid < depth * height * width; ++gid) {
        int im_d = gid / (width * height);
        int itmp = gid % (width * height);
        int im_h = itmp / width;
        int im_w = itmp % width;

        im_d += pad_d;
        im_h += pad_h;
        im_w += pad_w;

        int start_d = std::max(0, (im_d - dilation_d * (wei_d - 1) - 1) / stride_d + 1);
        int end_d = std::min(col_d, im_d / stride_d + 1);

        int start_h = std::max(0, (im_h - dilation_h * (wei_h - 1) - 1) / stride_h + 1);
        int end_h = std::min(col_h, im_h / stride_h + 1);

        int start_w = std::max(0, (im_w - dilation_w * (wei_w - 1) - 1) / stride_w + 1);
        int end_w = std::min(col_w, im_w / stride_w + 1);

        _FLOAT_ACCUM tmp = (_FLOAT_ACCUM)0;

        for (int cz = start_d; cz < end_d; ++cz) {
            for (int cy = start_h; cy < end_h; ++cy) {
                for (int cx = start_w; cx < end_w; ++cx) {
                    if ((im_d - cz * stride_d) % dilation_d == 0 &&
                        (im_h - cy * stride_h) % dilation_h == 0 &&
                        (im_w - cx * stride_w) % dilation_w == 0) {
                        int z = (im_d - cz * stride_d) / dilation_d;
                        int y = (im_h - cy * stride_h) / dilation_h;
                        int x = (im_w - cx * stride_w) / dilation_w;

                        int col_off =
                            (((((z * wei_h) + y) * wei_w + x) * col_d + cz) * col_h + cy) * col_w + cx;

                        tmp += col[col_off];
                    }
                }
            }
        }
        im[gid + im_offset] = tmp;
    }
}

int main(int argc, char *argv[]) {
  
    if (argc != 20) {
        std::cerr << "Usage: " << argv[0] << " <depth> <height> <width> <col_d> <col_h> <col_w> "
                    << "<wei_d> <wei_h> <wei_w> <pad_d> <pad_h> <pad_w> "
                    << "<stride_d> <stride_h> <stride_w> <dilation_d> <dilation_h> <dilation_w> <im_offset>\n";
        return -1;
    }

    const int depth = std::stoi(argv[1]);
    const int height = std::stoi(argv[2]);
    const int width = std::stoi(argv[3]);
    const int col_d = std::stoi(argv[4]);
    const int col_h = std::stoi(argv[5]);
    const int col_w = std::stoi(argv[6]);
    const int wei_d = std::stoi(argv[7]);
    const int wei_h = std::stoi(argv[8]);
    const int wei_w = std::stoi(argv[9]);
    const int pad_d = std::stoi(argv[10]);
    const int pad_h = std::stoi(argv[11]);
    const int pad_w = std::stoi(argv[12]);
    const int stride_d = std::stoi(argv[13]);
    const int stride_h = std::stoi(argv[14]);
    const int stride_w = std::stoi(argv[15]);
    const int dilation_d = std::stoi(argv[16]);
    const int dilation_h = std::stoi(argv[17]);
    const int dilation_w = std::stoi(argv[18]);
    const int im_offset = std::atoi(argv[19]);

    // Define other necessary variables and parameters (e.g., col_d, col_h, col_w, etc.)

    size_t size_im = depth * height * width * sizeof(_FLOAT);
    size_t size_col = col_d * col_h * col_w * sizeof(_FLOAT); // Example calculation, adjust as needed

    // Allocate host memory
    _FLOAT *h_im, *h_col ;
    h_im = (_FLOAT *)malloc(size_im);
    h_col = (_FLOAT *)malloc(size_col);

    // Generate random data 
    for (size_t i = 0; i < col_d * col_h * col_w; i++) {
        h_col[i] = static_cast<_FLOAT>(rand()) / RAND_MAX;
    }

    for (size_t i = 0; i < depth * height * width; i++) {
        h_im[i] = 0; 
    }

    // Allocate device memory
    _FLOAT *d_im, *d_col;

    CHECK(hipMalloc(&d_im, size_im));
    CHECK(hipMalloc(&d_col, size_col));

    // Copy data from host to device
    CHECK(hipMemcpy(d_im, h_im, size_im, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_col, h_col, size_col, hipMemcpyHostToDevice));

    // Define blockDim and gridDim
    dim3 blockDim(256); // for example, 256 threads per block
    dim3 gridDim((depth * height * width + blockDim.x - 1) / blockDim.x); // Ensure all pixels are covered

    // Launch the kernel
    Col2Im3d<<<gridDim, blockDim>>>(d_col, col_d, col_h, col_w, wei_d, wei_h, wei_w, 
                                    pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, 
                                    dilation_d, dilation_h, dilation_w, depth, height, width, 
                                    d_im, /*im_offset*/ 0);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "hip Error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    CHECK(hipMemcpy(h_im, d_im, size_im, hipMemcpyDeviceToHost));

    // Process the results on the host
    // Allocate memory for reference result
    float* h_im_ref = (float*)malloc(size_im);
    Col2Im3d_host( h_col, col_d, col_h, col_w, wei_d, wei_h, wei_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, dilation_d,
                   dilation_h, dilation_w, depth, height, width, h_im_ref , im_offset);


    // Compare GPU results with the reference result
    float maxError = 0.0;
    for (size_t i = 0; i < height * width; ++i) {
        maxError = std::max(maxError, std::fabs(h_im_ref[i] - h_im[i]));
    }
    std::cout << "Maximum error: " << maxError << std::endl;


    // Cleanup
    delete[] h_im;
    delete[] h_col;
    delete[] h_im_ref;
    hipFree(d_im);
    hipFree(d_col);

    return 0;
}
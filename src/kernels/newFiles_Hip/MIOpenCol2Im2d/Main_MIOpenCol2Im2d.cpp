// #include "float_types.h"
#include <hip/hip_runtime.h>
#include "MIOpenCol2Im2d.cpp"

#include <iostream>
#include <vector>

#define CHECK(error)                                      \
    if (error != hipSuccess) {                            \
        std::cerr << "HIP error: " << hipGetErrorString(error) << std::endl; \
        exit(1);                                           \
    }


void Col2Im2d_host(_FLOAT* col,
                  const int col_h,
                  const int col_w,
                  const int wei_h,
                  const int wei_w,
                  const int pad_h,
                  const int pad_w,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_h,
                  const int dilation_w,
                  const int height,
                  const int width,
                  _FLOAT* im,
                  const int im_offset) {
    // Initialize the output image array
    for (int i = 0; i < height * width; ++i) {
        im[i + im_offset] = 0;
    }

    int im_ch, im_pix, im_h, im_w, start_h, end_h, start_w, end_w, ch_offset, col_off_y, col_off_x ;

    for (int gid = 0; gid < height * width ; ++gid) {
        im_ch = gid / (width * height);
        im_pix = gid % (width * height);
        im_h = (im_pix / width) + pad_h;
        im_w = (im_pix % width) + pad_w;


        if(im_h < dilation_h * (wei_h - 1) + 1)
            start_h = 0 ;
        else
            start_h = (im_h - dilation_h * (wei_h - 1) - 1) / stride_h + 1;

        end_h = std::min(col_h, im_h / stride_h + 1);

        if(im_w < dilation_w * (wei_w - 1) + 1)
            start_w  = 0;
        else     
            start_w = (im_w - dilation_w * (wei_w - 1) - 1) / stride_w + 1;

        end_w = std::min(col_w, im_w / stride_w + 1);

        ch_offset = im_ch * col_w * col_h * wei_w * wei_h;
        col += ch_offset;

        _FLOAT_ACCUM tmp = (_FLOAT_ACCUM)0;

        for (int cy = start_h; cy < end_h; ++cy) {
            for (int cx = start_w; cx < end_w; ++cx) {
                if ((im_h - cy * stride_h) % dilation_h == 0 && 
                    (im_w - cx * stride_w) % dilation_w == 0) {
                    col_off_y = cy + (((im_h - cy * stride_h) / dilation_h) * wei_w * col_h);
                    col_off_x = cx + (((im_w - cx * stride_w) / dilation_w) * col_w * col_h);

                    tmp += col[col_off_y * col_w + col_off_x];
                }
            }
        }
        im[gid + im_offset] = static_cast<FLOAT>(tmp);
    }
}

int main(int argc, char *argv[]) {

    if (argc != 14) {
        std::cerr << "Usage: " << argv[0] << " [col_h] [col_w] [wei_h] [wei_w] [pad_h] [pad_w] [stride_h] [stride_w] [dilation_h] [dilation_w] [height] [width] [im_offset] \n";
        return 1;
    }


     // Define the dimensions and other parameters
    const int col_h = std::atoi(argv[1]);
    const int col_w = std::atoi(argv[2]);
    const int wei_h = std::atoi(argv[3]);
    const int wei_w = std::atoi(argv[4]);
    const int pad_h = std::atoi(argv[5]);
    const int pad_w = std::atoi(argv[6]);
    const int stride_h = std::atoi(argv[7]);
    const int stride_w = std::atoi(argv[8]);
    const int dilation_h = std::atoi(argv[9]);
    const int dilation_w = std::atoi(argv[10]);
    const int height = std::atoi(argv[11]);
    const int width = std::atoi(argv[12]);
    const int im_offset = std::atoi(argv[13]);
       
   

    // Allocate host memory
    _FLOAT *h_col, *h_im;
    size_t col_size = col_h * col_w * sizeof(_FLOAT);
    size_t im_size = height * width * sizeof(_FLOAT);

    h_col = (_FLOAT *)malloc(col_size);
    h_im = (_FLOAT *)malloc(im_size);

    // Initialize host memory
    for (size_t i = 0; i < col_h * col_w; i++) {
        h_col[i] = static_cast<_FLOAT>(rand()) / RAND_MAX;
    }

    for (size_t i = 0; i < height * width; i++) {
        h_im[i] = static_cast<_FLOAT>(rand()) / RAND_MAX; 
    }
   

    // Allocate device memory
    _FLOAT *d_col, *d_im;

    CHECK(hipMalloc((void **)&d_col, col_size));
    CHECK(hipMalloc((void **)&d_im, im_size));

    // Copy data from host to device
    CHECK(hipMemcpy(d_col, h_col, col_size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_im, h_im, im_size, hipMemcpyHostToDevice));

    // // Kernel launch parameters
    dim3 blockDim(256);  // Adjust as necessary
    dim3 gridDim((height * width + blockDim.x - 1) / blockDim.x);

    // dim3 blockDim(16, 16);
    // dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    Col2Im2d<<<gridDim, blockDim>>>(d_col, col_h, col_w, wei_h, wei_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, height, width, d_im, im_offset);

    // Check for errors
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        fprintf(stderr, "hip Error: %s\n", hipGetErrorString(error));
        return -1;
    }

    // Copy results back to host
    CHECK(hipMemcpy(h_im, d_im, im_size, hipMemcpyDeviceToHost));

    // Process the results on the host
    // Allocate memory for reference result
    float* h_im_ref = (float*)malloc(im_size);
    // Generate reference result
    Col2Im2d_host(h_col, col_h, col_w, wei_h, wei_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, height, width, h_im_ref, im_offset);

    // Compare GPU results with the reference result
    float maxError = 0.0;
    for (size_t i = 0; i < height * width; ++i) {
        maxError = std::max(maxError, std::fabs(h_im_ref[i] - h_im[i]));
    }
    std::cout << "Maximum error: " << maxError << std::endl;

    // Free device memory
    hipFree(d_col);
    hipFree(d_im);

    // Free host memory
    free(h_col);
    free(h_im);
    free(h_im_ref) ;

    // delete[] h_im ;
    // delete[] h_col ;
    // delete[] h_im_ref ;

    return 0;
}
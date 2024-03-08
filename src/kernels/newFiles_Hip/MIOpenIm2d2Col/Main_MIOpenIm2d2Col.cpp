#include <hip/hip_runtime.h>
#include "MIOpenIm2d2Col.cpp"

#include <iostream>
#include <vector>

#define CHECK(error)                                      \
    if (error != hipSuccess) {                            \
        std::cerr << "HIP error: " << hipGetErrorString(error) << std::endl; \
        exit(1);                                           \
    }



void Im2d2Col_host(const int data_size_off,
                  data_t* im,
                  const int im_offset,
                  const int h,
                  const int w,
                  const int wei_h,
                  const int wei_w,
                  const int out_h,
                  const int out_w,
                  const int pad_h,
                  const int pad_w,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_h,
                  const int dilation_w,
                  data_t* col,
                  const int num_ch_per_wg,
                  const int num_im_blks_x,
                  const int num_im_blks,
                  const int tile_sz_x,
                  const int tile_sz_y) {
    // Assuming data_t is a type alias for a floating-point type like float or double

    data_t* im_off = im + im_offset;

    for (int gid = 0; gid < num_ch_per_wg * num_im_blks_x * num_im_blks; ++gid) {
        int wg_ch = gid / num_im_blks;
        int im_x = ((gid % num_im_blks) % num_im_blks_x) * tile_sz_x;
        int im_y = ((gid % num_im_blks) / num_im_blks_x) * tile_sz_y;

        int out_cols_wg = std::min(tile_sz_x, out_w - im_x);
        int out_rows_wg = std::min(tile_sz_y, out_h - im_y);
        int im_cols_wg = (tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1;

        for (int out_y = 0; out_y < out_rows_wg; ++out_y) {
            for (int out_x = 0; out_x < out_cols_wg; ++out_x) {
                int col_x = (im_y + out_y) * out_w + im_x + out_x;
                int col_y = wg_ch * out_h * out_w * wei_h * wei_w;

                for (int y = 0; y < wei_h; ++y) {
                    for (int x = 0; x < wei_w; ++x) {
                        int im_off_h = out_y * stride_h + y * dilation_h;
                        int im_off_w = out_x * stride_w + x * dilation_w;

                        index_t col_off = col_y + col_x + (y * wei_w + x) * out_h * out_w;

                        if (im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w) {
                            col[col_off] = im_off[(im_off_h)*w + im_off_w];
                        } else {
                            col[col_off] = 0;
                        }
                    }
                }
            }
        }
    }
}

// Function to validate the results of the GPU against the CPU
bool validateResults(data_t* cpu_result, data_t* gpu_result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(cpu_result[i] - gpu_result[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}


int main(int argc, char *argv[]) {
    if (argc != 20) {
        std::cerr << "Usage: " << argv[0] << " [parameters...]" << std::endl;
        return -1;
    }

    int data_size_off = std::stoi(argv[1]);
    int im_offset = std::stoi(argv[2]);
    int h = std::stoi(argv[3]), w = std::stoi(argv[4]);
    int wei_h = std::stoi(argv[5]), wei_w = std::stoi(argv[6]);
    int out_h = std::stoi(argv[7]), out_w = std::stoi(argv[8]);
    int pad_h = std::stoi(argv[9]), pad_w = std::stoi(argv[10]);
    int stride_h = std::stoi(argv[11]), stride_w = std::stoi(argv[12]);
    int dilation_h = std::stoi(argv[13]), dilation_w = std::stoi(argv[14]);
    int num_ch_per_wg = std::stoi(argv[15]);
    int num_im_blks_x = std::stoi(argv[16]), num_im_blks = std::stoi(argv[17]);
    int tile_sz_x = std::stoi(argv[18]), tile_sz_y = std::stoi(argv[19]);
    

    // Allocate host memory
    size_t size_im = ; 
    size_t size_col = ;

    data_t *h_im, *h_col, *h_col_cpu ;

    h_im = (data_t*)malloc(size_im);
    h_col =(data_t*)malloc(size_col);

    // Generate random data 
    for (size_t i = 0; i < ; i++) {
        h_col[i] = static_cast<data_t>(rand()) / RAND_MAX;
    }

    for (size_t i = 0; i < ; i++) {
        h_im[i] = static_cast<data_t>(rand()) / RAND_MAX; 
    }


    // Allocate device memory
    data_t *d_im, *d_col;
    CHECK(hipMalloc(&d_im, size_im * sizeof(data_t)));
    CHECK(hipMalloc(&d_col, size_col * sizeof(data_t)));

    // Copy data from host to device
    CHECK(hipMemcpy(d_im, h_im, size_im * sizeof(data_t), hipMemcpyHostToDevice));




    // Define blockDim and gridDim for the GPU kernel
    dim3 blockDim(256);
    dim3 gridDim((size_col + blockDim.x - 1) / blockDim.x);

    // Launch the GPU kernel
    Im2d2Col<<<gridDim, blockDim>>>(data_size_off, d_im, im_offset, h, w, wei_h, wei_w, out_h, out_w, 
                                    pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
                                    d_col, num_ch_per_wg, num_im_blks_x, num_im_blks, tile_sz_x, tile_sz_y);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "hip Error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host from GPU
    CHECK(hipMemcpy(h_col, d_col, size_col * sizeof(data_t), hipMemcpyDeviceToHost));

    h_col_cpu = (data_t*)malloc(size_col);
    // Call the CPU function
    Im2d2Col_host(data_size_off,
                    im,
                    im_offset,
                    h,
                    w,
                    wei_h,
                    wei_w,
                    out_h,
                    out_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    h_col_cpu,
                    num_ch_per_wg,
                    num_im_blks_x,
                    num_im_blks,
                    tile_sz_x,
                    tile_sz_y);

    // Validate the results
    if (validateResults(h_col_cpu, h_col, size_col)) {
        std::cout << "Validation successful: CPU and GPU results match." << std::endl;
    } else {
        std::cerr << "Validation failed: CPU and GPU results do not match." << std::endl;
    }

    // Cleanup
    delete[] h_im;
    delete[] h_col;
    delete[] h_col_cpu;
    
    hipFree(d_im);
    hipFree(d_col);

    return 0;
}
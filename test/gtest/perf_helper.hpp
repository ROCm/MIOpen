/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include "get_handle.hpp"

#define NUM_PERF_RUNS 5

struct PerfHelper
{

    template <typename T>
    static T perf_min(const std::vector<T>& data)
    {
        if(data.empty())
            throw std::invalid_argument("Empty vector");
        return *std::min_element(data.begin(), data.end());
    }

    template <typename T>
    static T perf_max(const std::vector<T>& data)
    {
        if(data.empty())
            throw std::invalid_argument("Empty vector");
        return *std::max_element(data.begin(), data.end());
    }

    template <typename T>
    static double perf_mean(const std::vector<T>& data)
    {
        if(data.empty())
            throw std::invalid_argument("Empty vector");
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    template <typename T>
    static double perf_median(std::vector<T> data)
    {
        if(data.empty())
            throw std::invalid_argument("Empty vector");
        size_t size = data.size();
        std::sort(data.begin(), data.end());
        if(size % 2 == 0)
        {
            return (data[size / 2 - 1] + data[size / 2]) / 2.0;
        }
        else
        {
            return data[size / 2];
        }
    }

    template <typename T>
    static double perf_standardDeviation(const std::vector<T>& data)
    {
        if(data.empty())
            throw std::invalid_argument("Empty vector");
        double data_mean = perf_mean(data);
        double sq_sum    = std::inner_product(
            data.begin(), data.end(), data.begin(), 0.0, std::plus<>(), [data_mean](T a, T b) {
                return (a - data_mean) * (b - data_mean);
            });
        return std::sqrt(sq_sum / data.size());
    }

    template <typename T>
    static std::tuple<T, T, double, double, double> calcStats(const std::vector<T>& data)
    {
        T min_val         = perf_min(data);
        T max_val         = perf_max(data);
        double mean_val   = perf_mean(data);
        double median_val = perf_median(data); // Note: This modifies the data(sorts it)
        double sd_val     = perf_standardDeviation(data);
        return {min_val, max_val, mean_val, median_val, sd_val};
    }

    template <typename T>
    static void
    writeStatsToCSV(const std::string& filename, const std::vector<T>& data, bool append)
    {
        std::ofstream file;
        file.open(filename, std::ios::app);

        if(!file.is_open())
        {
            throw std::runtime_error("Failed to open file");
        }

        std::tuple<T, T, double, double, double> stats = calcStats(data);
        file << (append ? "" : ",") << std::get<0>(stats) << "," << std::get<1>(stats) << ","
             << std::get<2>(stats) << "," << std::get<3>(stats) << "," << std::get<4>(stats)
             << (append ? "" : "\n");

        file.close();
    }

    static void writeHeaderToCSV(const std::string& filename)
    {
        std::ofstream file;

        // If the file already exists, do not write the header
        if(!std::filesystem::exists(filename))
        {
            file.open(filename, std::ios::app);
            if(!file.is_open())
            {
                throw std::runtime_error("Failed to open file");
            }
            file
                << "OCL_Min,OCL_Max,OCL_Mean,OCL_Median,OCL_SD,HIP_Min,HIP_Max,HIP_Mean,HIP_Median,"
                   "HIP_SD\n";
            file.close();
        }
    }

    template <typename... Args>
    static void perfTest(miopen::Handle& handle,
                         const std::string& kernel_name,
                         const std::string& network_config,
                         const std::string& perf_filename_csv,
                         bool append,
                         Args&&... args)
    {
        // Get kernels matching the kernel_name and network_config from the cache
        auto&& kernels = handle.GetKernels(kernel_name, network_config);
        // Ensure we have at least one kernel
        assert(!kernels.empty());
        // Get the type of the elapsed time
        using elapsed_t = decltype(handle.GetKernelTime());
        // Vector to store elapsed times
        std::vector<elapsed_t> elapsedTime_ms;
        // Enable profiling
        handle.EnableProfiling();

        for(size_t i = 0; i < NUM_PERF_RUNS; i++)
        {
            // Execute the kernel
            kernels.front()(std::forward<Args>(args)...);
            // Append the elapsed time to the vector
            elapsedTime_ms.push_back(handle.GetKernelTime());
        }

        // Write the stats to the CSV file
        writeStatsToCSV(perf_filename_csv, elapsedTime_ms, !append);
    }
};

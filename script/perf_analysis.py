#!/usr/bin/env python3

import argparse
import os
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MIOpen test bench results.")
    # Add variable length argument for the results file names
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory for the CSV files.")
    parser.add_argument("--files", nargs="+", help="Kernel times file names to analyze.")
    parser.add_argument("--labels", nargs="+", help="Labels for the kernel time files.")

    args, _ = parser.parse_known_args()
    
    return args

def read_csv_file(file_path):
    """
    Read a CSV file and return the data as a list of dictionaries.
    Each dictionary corresponds to a row in the CSV file.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            headers = lines[0].strip().split(',')
            data = []
            for line in lines[1:]:
                values = [value.strip() for value in list(csv.reader([line.strip()]))[0]]
                data.append(dict(zip(headers, values)))
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def calculate_statistics(data):
    kernel_times = []
    inference_times = []
    failed_cases = 0
    for row in data:
      try:
        inference_time = float(row[' KTN_inference_time_ms'])
        if inference_time > 0:
          kernel_times.append(float(row[' Elapsed_GPU_time_average_ms']))
          inference_times.append(inference_time)
        else:
           failed_cases += 1
      except ValueError:
        failed_cases += 1

    kernel_times = np.array(kernel_times, dtype=float)
    inference_times = np.array(inference_times, dtype=float)
    stats = {
        'kernel_time_mean': np.mean(kernel_times),
        'kernel_time_median': np.median(kernel_times),
        'kernel_time_max': np.max(kernel_times),
        'inference_time_mean': np.mean(inference_times),
        'inference_time_median': np.median(inference_times),
        'inference_time_max': np.max(inference_times),
        'inference_time_min': np.std(inference_times),
        'failed_cases': failed_cases,
        'total_cases': len(data)
    }
    return stats

def run_worst_cases_analysis(results, top_number=10):
    """
    Analyze the worst cases based on kernel times and inference times.
    """
    
    # For each label, find the top worst cases based on kernel times.
    # Then, compare the performance from other labels in these cases.
    worst_cases = {}
    all_kernel_times = {}
    repeated_configs = {}
    for label, data in results.items():
        kernel_times = {}
        for row in data:
            try:
                inference_time = float(row[' KTN_inference_time_ms'])
                if inference_time > 0:
                    configuration = row['Configuration']
                    time = float(row[' Elapsed_GPU_time_average_ms'])
                    if configuration not in kernel_times:
                        kernel_times[configuration] = time
                    else:
                        raise LookupError(f"Configuration {configuration} appears multiple times in {label}.")
            except ValueError:
                continue

        # Get top_number worst cases based on kernel times
        sorted_cases = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)[:top_number]
        kernel_times_array = np.array(list(kernel_times.values()), dtype=float)
        print(f"max kernel time for {label}: {np.max(kernel_times_array):.2f} ms")
        worst_cases[label] = {
            'kernel_times': [time for _, time in sorted_cases],
            'configurations': [config for config, _ in sorted_cases]
        }
        all_kernel_times[label] = kernel_times

    for label, worst_cases_data in worst_cases.items():
        print(f"Worst cases for {label}:")
        # Find the corresponding cases in other labels
        for i, (config, time) in enumerate(zip(worst_cases_data['configurations'], worst_cases_data['kernel_times'])):
            print(f"  {i+1}. Configuration: {config}")
            for other_label, other_kernel_times in all_kernel_times.items():
                if config in other_kernel_times:
                    print(f"    {other_label}: {other_kernel_times[config]:.2f} ms")

def main():
    base_dir, kernel_time_files, kernel_time_labels = parse_args().base_dir, parse_args().files, parse_args().labels

    # Check if the base directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"The base directory {base_dir} does not exist.")

    # Check if the number of files and labels match
    if len(kernel_time_files) != len(kernel_time_labels):
        raise ValueError("The number of kernel time files must match the number of labels.")

    # Create a list to hold the file paths
    file_paths = [os.path.join(base_dir, f) for f in kernel_time_files]

    results = {}
    for path, label in zip(file_paths, kernel_time_labels):
        res = read_csv_file(path)
        results[label] = res

    statistics = {}
    for label, data in results.items():
      stats = calculate_statistics(data)
      statistics[label] = stats
      print(f"Statistics for {label}: {stats}")
      print()

    run_worst_cases_analysis(results)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
    
#!/bin/bash

# Add jq dependency check
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq."
    exit 1
fi

# If script is excuted with --help or without any arguments, print usage and exit.
if [[ "$*" == *"--help"* || "$*" == "" ]]; then
    echo "Usage: $0 --config <name> [OPTIONS]"
    echo "Options:"
    echo "  --log-level <level>: Set log level (optional, default: 5)"
    echo "  --gpu-id <device id>: Set GPU ID (optional, default: 0)"
    echo "  --algorithms <alg1>, <alg2>, ...: Set which algorithms to use (optional, default: ALL), options are FFT, DIRECT, WINOGRAD, GEMM, IMPLICIT_GEMM, or ALL."
    echo "  --tuning-db-dir <dir path>: Set tuning DB directory path (required for --incremental-tuning and --exhaustive-tuning flags)"
    echo "  --incremental-tuning: Use incremental tuning (optional)"
    echo "  --exhaustive-tuning: Use exhaustive tuning (optional)"
    echo "  --config-file <file>: Specify the configuration file (optional, default: configs.json)"
    echo "  --config <name>: Specify the configuration name to use from the config file (required)"
    echo "  --disable-kernel-cache: Disable kernel cache (optional)"
    echo "  --help: Show this help message"
    exit 0
fi

# Check that we didn't receive any flags that we don't recognize.
valid_flags="--log-level --gpu-id --algorithms --tuning-db-dir --incremental-tuning --exhaustive-tuning --config --config-file --help --disable-kernel-cache"
for arg in "$@"; do
    # Only check arguments that start with --
    if [[ "$arg" == --* ]]; then
        valid_flag=false
        for valid_arg in $valid_flags; do
            if [[ "$arg" == "$valid_arg" || "$arg" == "$valid_arg="* || "$arg" == "$valid_arg "* ]]; then
                valid_flag=true
                break
            fi
        done
        if [[ "$valid_flag" == false ]]; then
            echo "Error: Unrecognized flag: $arg"
            echo "Use --help for usage information."
            exit 1
        fi
    fi
done

config_file=""
if [[ "$*" == *"--config-file"* ]]; then
    config_file=$(echo "$*" | grep -oP '(?<=--config-file )\S+')
    echo "Using configuration file: $config_file"
else
    config_file="configs.json"
    echo "Using default configuration file: $config_file"
fi

# Check if config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file not found: $config_file"
    exit 1
fi

# Tuning DB path flag --tuning-db
if [[ "$*" == *"--tuning-db-dir"* ]]; then
    tuning_db=$(echo "$*" | grep -oP '(?<=--tuning-db-dir )\S+')
    echo "Using tuning DB: $tuning_db"
    # Create the tuning DB directory if it doesn't exist
    if [[ ! -d "$tuning_db" ]]; then
        echo "Creating tuning DB directory: $tuning_db"
        mkdir -p "$tuning_db"
    fi
    export MIOPEN_USER_DB_PATH="$tuning_db"
fi

# Check flag --disable-kernel-cache
if [[ "$*" == *"--disable-kernel-cache"* ]]; then
    echo "Disabling kernel cache"
    export MIOPEN_DISABLE_CACHE=1
else
    echo "Kernel cache is enabled by default"
fi

tuning="_"
# Increamental tuning flag --incremental-tuning
if [[ "$*" == *"--incremental-tuning"* ]]; then
    echo "Running with incremental tuning"
    export MIOPEN_FIND_MODE=3
    export MIOPEN_FIND_ENFORCE=3

    # Check if the tuning DB path is provided
    if [[ -z "$tuning_db" ]]; then
        echo "Error: Tuning DB path must be provided with --tuning-db when using --incremental-tuning" >&2
        exit 1
    fi

    # Check that kernel cache is not disabled
    if [[ "$*" == *"--disable-kernel-cache"* ]]; then
        echo "Error: Kernel cache must be enabled when using --incremental-tuning" >&2
        exit 1
    fi

    tuning="_incremental-tuning_"
fi

# Exhaustive tuning flag --exhaustive-tuning
if [[ "$*" == *"--exhaustive-tuning"* ]]; then
    echo "Running with exhaustive tuning"
    export MIOPEN_FIND_MODE=3
    export MIOPEN_FIND_ENFORCE=3

    # Check if the tuning DB path is provided
    if [[ -z "$tuning_db" ]]; then
        echo "Error: Tuning DB path must be provided with --tuning-db when using --exhaustive-tuning" >&2
        exit 1
    fi

    # Check that kernel cache is not disabled
    if [[ "$*" == *"--disable-kernel-cache"* ]]; then
        echo "Error: Kernel cache must be enabled when using --exhaustive-tuning" >&2
        exit 1
    fi

    export MIOPEN_SYSTEM_DB_PATH="$MIOPEN_USER_DB_PATH"
    tuning="_exhaustive-tuning_"
fi

# Check log level flag
if [[ "$*" == *"--log-level"* ]]; then
    log_level=$(echo "$*" | grep -oP '(?<=--log-level )\d+')
    echo "Using log level: $log_level"
else
    log_level=5
    echo "Using default log level: $log_level"
fi

# Check if flag --gpu-id is given
if [[ "$*" == *"--gpu-id"* ]]; then
    gpu_id=$(echo "$*" | grep -oP '(?<=--gpu-id )\d+')
    echo "Using GPU ID: $gpu_id"
else
    gpu_id=0
    echo "Using default GPU ID: $gpu_id"
fi
export HIP_VISIBLE_DEVICES=$gpu_id


# Flag --algorithm can take values "ALL", "FFT", "DIRECT", "WINOGRAD", "GEMM", and "IMPLICIT_GEMM"
# Multiple algorithms can be specified as a comma-separated list.
# If no algorithm is specified, default to "ALL"
if [[ "$*" == *"--algorithms"* ]]; then
    algorithms_raw=$(echo "$*" | sed -n 's/.*--algorithms\s\+\([^-]*\).*/\1/p' | xargs)
algorithms=$(echo "$algorithms_raw" | tr -s ' ' | sed 's/ *, */,/g')
    echo "Using algorithms: $algorithms"
else
    algorithms="ALL"
    echo "Using default algorithms: $algorithms"
fi

# Set the algorithm environment variables based on the selected algorithms
export MIOPEN_DEBUG_CONV_FFT=0
export MIOPEN_DEBUG_CONV_DIRECT=0
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DEBUG_CONV_GEMM=0
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
algs=""
if [[ "$algorithms" == *"ALL"* ]]; then
    export MIOPEN_DEBUG_CONV_FFT=1
    export MIOPEN_DEBUG_CONV_DIRECT=1
    export MIOPEN_DEBUG_CONV_WINOGRAD=1
    export MIOPEN_DEBUG_CONV_GEMM=1
    export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
    algs="ALL"
else
    IFS=',' read -r -a algorithm_array <<< "$algorithms"
    for algorithm in "${algorithm_array[@]}"; do
        case $algorithm in
            "FFT")
                export MIOPEN_DEBUG_CONV_FFT=1
                algs+="FFT-"
                ;;
            "DIRECT")
                export MIOPEN_DEBUG_CONV_DIRECT=1
                algs+="DIRECT-"
                ;;
            "WINOGRAD")
                export MIOPEN_DEBUG_CONV_WINOGRAD=1
                algs+="WINOGRAD-"
                ;;
            "GEMM")
                export MIOPEN_DEBUG_CONV_GEMM=1
                algs+="GEMM-"
                ;;
            "IMPLICIT_GEMM")
                export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
                algs+="IMPLICIT_GEMM-"
                ;;
            *)
                echo "Error: Unknown algorithm: $algorithm" >&2
                exit 1
                ;;
        esac
    done
fi 

# Strip the trailing dash from algs.
if [[ $algs == *- ]]; then
    algs=${algs::-1}
fi

# Debug logging flags.
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_LOG_LEVEL=$log_level
export MIOPEN_ENABLE_LOGGING_ELAPSED_TIME=1

pid=$$
miopen_dirver_path="$PWD/../build/bin/MIOpenDriver"

# Create a CSV file with headers to record the kernel times.
kernel_times_path="../logs/kernel_times_${algs}${tuning}${pid}.csv"
echo "Configuration, Algo_num, Solution_ID, Solver_name, Elapsed_GPU_time_average_ms, KTN_inference_time_ms" > "$kernel_times_path"

# Load configuration parameters
config_name="default"
if [[ "$*" == *"--config"* ]]; then
    config_name=$(echo "$*" | grep -oP '(?<=--config )\S+')
fi
echo "Using configuration: $config_name"
if ! jq -e ".$config_name" "$config_file" > /dev/null; then
    echo "Error: Configuration '$config_name' not found in $config_file"
    exit 1
fi

# Extract parameters from JSON
get_param() {
    local param=$1
    local default=$2
    value=$(jq -r ".$config_name.$param // .$default.$param // \"$default\"" "$config_file")
    echo "$value"
}

# Get parameters with fallback to default config
verb=$(get_param "verb" "default")
batch_size=$(get_param "batch_size" "default")
in_channels=$(get_param "in_channels" "default")
in_depth=$(get_param "in_depth" "default")
in_height=$(get_param "in_height" "default")
in_width=$(get_param "in_width" "default")
filters=$(get_param "filters" "default")
filter_depth=$(get_param "filter_depth" "default") 
filter_height=$(get_param "filter_height" "default")
filter_width=$(get_param "filter_width" "default")
pad_depth=$(get_param "pad_depth" "default")
pad_height=$(get_param "pad_height" "default")
pad_width=$(get_param "pad_width" "default")
stride_depth=$(get_param "stride_depth" "default")
stride_height=$(get_param "stride_height" "default")
stride_width=$(get_param "stride_width" "default")
dilation_depth=$(get_param "dilation_depth" "default")
dilation_height=$(get_param "dilation_height" "default")
dilation_width=$(get_param "dilation_width" "default")
spatial_dim=$(get_param "spatial_dim" "default")
in_layout=$(get_param "in_layout" "default")
fil_layout=$(get_param "fil_layout" "default")
out_layout=$(get_param "out_layout" "default")
mode=$(get_param "mode" "default")
group_count=$(get_param "group_count" "default")
forw=$(get_param "forw" "default")
verify=$(get_param "verify" "default")
wall=$(get_param "wall" "default")
time=$(get_param "time" "default")

# Check whether the JSON config has any environemtn variables to set. 
# They will override any environment variables set in the shell.
set_env_vars_from_json() {
    # Get environment variables from config
    env_vars_json=$(jq -r ".$config_name.env_vars // .default.env_vars // {}" "$config_file")
    
    if [ "$env_vars_json" != "null" ] && [ "$env_vars_json" != "{}" ]; then
        while IFS="=" read -r key value; do
            # Skip empty lines
            [ -z "$key" ] && continue
            # Export the environment variable
            export "$key"="$value"
        done < <(jq -r ".$config_name.env_vars // .default.env_vars // {} | to_entries[] | \"\(.key)=\(.value)\"" "$config_file")
    fi
}
set_env_vars_from_json

log_path="../logs/${mode}_${algs}_${config_name}${tuning}${pid}.log"

# Time the MIOpen driver call and log the output to the log file.
echo "Log file: $log_path"
mkdir -p $(dirname "$log_path")

echo "=== Execution started at $(date) ===" > "$log_path"
echo "" >> "$log_path"
{
    time ( 
        $miopen_dirver_path ${verb} \
            -n ${batch_size} -c ${in_channels} --in_d ${in_depth} -H ${in_height} -W ${in_width} \
            -k ${filters} --fil_d ${filter_depth} -y ${filter_height} -x ${filter_width} \
            --pad_d ${pad_depth} -p ${pad_height} -q ${pad_width} \
            --conv_stride_d ${stride_depth} -u ${stride_height} -v ${stride_width} \
            --dilation_d ${dilation_depth} -l ${dilation_height} -j ${dilation_width} \
            --spatial_dim ${spatial_dim} --in_layout ${in_layout} \
            --fil_layout ${fil_layout} --out_layout ${out_layout} \
            -m ${mode} -g ${group_count} -F ${forw} -V ${verify} \
            --wall ${wall} --time ${time} \
            >> "$log_path" 2>&1
        # Check if the MIOpenDriver command was successful
        if [ $? -ne 0 ]; then
            echo "ERROR: MIOpenDriver command failed." >> "$log_path"
        fi
        echo ""
        echo "=== MIOpenDriver execution time ===" >> "$log_path"
    ) 
} >> "$log_path" 2>&1
echo "" >> "$log_path"

# Record the environment variables and parameters used in the execution
echo "=== Environment Variables ===" >> "$log_path"
env >> "$log_path"
echo "=== Execution completed at $(date) ===" >> "$log_path"

# We want to extract from the log the following lines
# "GPU Kernel XXX Elapsed: Y ms (average)" where XXX is an uninteresting string or strings, and we want to
# extract the Y value.

extract_kernel_times() {
    local log_file="$1"
    local times_file="$2"
    local config="$3"
    
    local algo_num=""
    local sol_id=""
    local kernel_name=""
    local time_value=""
    local ktn_inference_time=""

    while IFS= read -r line; do
        if [[ $line =~ MIOpen\ (.*)\ Algorithm:\ ([0-9]+),\ Solution:\ ([0-9]+)/([A-Za-z0-9_]+) ]]; then
            algo_num="${BASH_REMATCH[2]}"
            sol_id="${BASH_REMATCH[3]}"
            kernel_name="${BASH_REMATCH[4]}"     
        elif [[ $line =~ GPU\ Kernel\ Time\ (.*)\ Elapsed:\ ([0-9.]+)\ ms ]]; then
            time_value="${BASH_REMATCH[2]}"
        elif [[ $line =~ \[ModelSetParams\]\ KTN\ ran\ for\ ([0-9]+)\ micro-seconds ]]; then
            ktn_micro="${BASH_REMATCH[1]}"
            ktn_inference_time=$(awk "BEGIN {printf \"%.3f\", $ktn_micro / 1000}")
        fi
    done < "$log_file"

    echo "$config, $algo_num, $sol_id, $kernel_name, $time_value, $ktn_inference_time" >> "$times_file"
}

extract_kernel_times "$log_path" "$kernel_times_path" "$config_name"
echo "Kernel times extracted to: $kernel_times_path"
echo "Execution completed."

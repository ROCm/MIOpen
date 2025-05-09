#!/bin/bash

# Add jq dependency check
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq."
    exit 1
fi

# If script is excuted with --help or without any arguments, print usage and exit.
if [[ "$*" == *"--help"* || "$*" == "" ]]; then
    echo "Usage: $0 [--local-build] [--log-level <level>] [--gpu-id <id>] [--num-batches <num>] [--algorithms <algorithms>]"
    echo "  --log-level: Set log level (default: 5)"
    echo "  --gpu-id: Set GPU ID (default: 0)"
    echo "  --algorithms: Set algorithms to use (default: ALL), options are FFT, DIRECT, WINOGRAD, GEMM, IMPLICIT_GEMM, or ALL"
    echo "  --tuning-db-dir: Set tuning DB directory path (required for --incremental-tuning and --exhaustive-tuning)"
    echo "  --incremental-tuning: Use incremental tuning"
    echo "  --exhaustive-tuning: Use exhaustive tuning"
    echo "  --config <name>: Specify the configuration name to use from the JSON config file"
    echo "  --config-file <file>: Specify the configuration file (default: configs.json)"
    echo "  --help: Show this help message"
    exit 0
fi

# Check that we didn't get any flags that we don't recognize
valid_flags="--local-build --log-level --gpu-id --num-batches --algorithms --tuning-db-dir --incremental-tuning --exhaustive-tuning --config --config-file --help"
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

# New flag for JSON config file and configuration name
config_name="default"
config_file=""

# Process command line arguments for config
if [[ "$*" == *"--config"* ]]; then
    config_name=$(echo "$*" | grep -oP '(?<=--config )\S+')
fi
echo "Using configuration: $config_name"

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

# Load configuration parameters
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

# Log file path,append the the log file name with selected algorithms and process ID
# to avoid overwriting.
pid=$$

log_path="../logs/${mode}_${algs}_${config_name}${tuning}${pid}.log"
miopen_dirver_path="$PWD/../build/bin/MIOpenDriver"

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
            -m ${mode} -g ${group_count} -F ${forw} -t ${verify} \
            >> "$log_path" 2>&1
        echo ""
        echo "=== MIOpenDriver execution time ===" >> "$log_path"
    ) 
} >> "$log_path" 2>&1
echo "" >> "$log_path"
echo "=== Execution completed at $(date) ===" >> "$log_path"
echo "Execution completed."

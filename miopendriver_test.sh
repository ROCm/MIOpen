#!/bin/bash

# Define default values
MIOPEN_DEFAULT_BACKEND="HIP"
MIOPEN_BACKEND="$MIOPEN_DEFAULT_BACKEND"
MIOPEN_EMBED_BUILD="Off"  # Default is Off

# Check if MIOPEN_BACKEND is overwritten by external flag
if [ -n "$1" ]; then
    MIOPEN_BACKEND="$1"
fi

# Check if MIOPEN_EMBED_BUILD is overwritten by external flag
if [ -n "$2" ]; then
    MIOPEN_EMBED_BUILD="$2"
fi

# Define set_var_to_condition macro
set_var_to_condition() {
    if [ $# -ne 0 ]; then
        eval "$1=true"
    else
        eval "$1=false"
    fi
}

# Set MIOPEN_BUILD_DRIVER_DEFAULT based on conditions
MIOPEN_BUILD_DRIVER_DEFAULT=true
if [ "$MIOPEN_EMBED_BUILD" = "Off" ] && [ "$MIOPEN_BACKEND" != "HIPNOGPU" ]; then
    MIOPEN_BUILD_DRIVER_DEFAULT=true
else
    MIOPEN_BUILD_DRIVER_DEFAULT=false
fi

# Function to compare environment variable to multiple possibilities
# Usage: compare_env_variable <variable_name> <value1> <value2> ... <valueN>
compare_env_variable() {
    variable_name=$1
    shift
    expected_values=("$@")
    
    # Get the value of the environment variable
    actual_value="${!variable_name}"
    
    # Check if the actual value matches any of the expected values
    for expected_value in "${expected_values[@]}"; do
        if [[ "${actual_value}" == "${expected_value}" ]]; then
            echo "true"
            return 0  # Return true
        fi
    done
    
    # If none of the expected values match, return false
    echo "false"
    return 1
}

# Determine input args based on options
if [ "$(compare_env_variable "MIOPEN_TEST_HALF" "1" "on" "On" "ON")" = true ]; then
    MIOPEN_TEST_FLOAT_ARG=--half
    MIOPENDRIVER_MODE_CONV=convfp16
    MIOPENDRIVER_MODE_POOL=poolfp16
    MIOPENDRIVER_MODE_BN=bnormfp16
    MIOPENDRIVER_MODE_GEMM=gemmfp16
elif [ "$(compare_env_variable "MIOPEN_TEST_INT8" "1" "on" "On" "ON")" = true ]; then
    MIOPEN_TEST_FLOAT_ARG=--int8
    MIOPENDRIVER_MODE_CONV=convint8
    MIOPENDRIVER_MODE_POOL=NOT_SUPPORTED
    MIOPENDRIVER_MODE_BN=NOT_SUPPORTED
    MIOPENDRIVER_MODE_GEMM=NOT_SUPPORTED
elif [ "$(compare_env_variable "MIOPEN_TEST_BFLOAT16" "1" "on" "On" "ON")" = true ]; then
    MIOPEN_TEST_FLOAT_ARG=--bfloat16
    MIOPENDRIVER_MODE_CONV=convbfp16
    MIOPENDRIVER_MODE_POOL=NOT_SUPPORTED
    MIOPENDRIVER_MODE_BN=NOT_SUPPORTED
    MIOPENDRIVER_MODE_GEMM=NOT_SUPPORTED
else
    MIOPEN_TEST_FLOAT_ARG=--float
    MIOPENDRIVER_MODE_CONV=conv
    MIOPENDRIVER_MODE_POOL=pool
    MIOPENDRIVER_MODE_BN=bnorm
    MIOPENDRIVER_MODE_GEMM=gemm
fi

# Detect GPU
function gpu_detection() 
{
    MIOPEN_TEST_GPU_DETECTION_FAILED=false
    MIOPEN_NO_GPU=false

    if [ ! \( "$MIOPEN_TEST_GFX900" -o "$MIOPEN_TEST_GFX906" -o "$MIOPEN_TEST_GFX908" -o "$MIOPEN_TEST_GFX90A" -o "$MIOPEN_TEST_GFX94X" -o "$MIOPEN_TEST_GFX103X" -o "$MIOPEN_TEST_GFX110X" -o "$MIOPEN_TEST_HIP_NOGPU" \) ]; then
        ROCMINFO=$(command -v rocminfo)
        if [ -n "$ROCMINFO" ]; then
            ROCMINFO_OUTPUT=$(rocminfo 2>/dev/null | grep -E 'gfx1030|gfx1031|gfx1100|gfx1101|gfx1102|gfx900|gfx906|gfx908|gfx90a|gfx94')
            ROCMINFO_EXIT_STATUS=$?

            if [ -n "$ROCMINFO_OUTPUT" ]; then
                if [[ $ROCMINFO_OUTPUT =~ "gfx1030" || $ROCMINFO_OUTPUT =~ "gfx1031" ]]; then
                    MIOPEN_TEST_GFX103X=true
                elif [[ $ROCMINFO_OUTPUT =~ "gfx1100" || $ROCMINFO_OUTPUT =~ "gfx1101" || $ROCMINFO_OUTPUT =~ "gfx1102" ]]; then
                    MIOPEN_TEST_GFX110X=true
                elif [[ $ROCMINFO_OUTPUT =~ "gfx900" ]]; then
                    MIOPEN_TEST_GFX900=true
                elif [[ $ROCMINFO_OUTPUT =~ "gfx906" ]]; then
                    MIOPEN_TEST_GFX906=true
                elif [[ $ROCMINFO_OUTPUT =~ "gfx908" ]]; then
                    MIOPEN_TEST_GFX908=true
                elif [[ $ROCMINFO_OUTPUT =~ "gfx90a" ]]; then
                    MIOPEN_TEST_GFX90A=true
                elif [[ $ROCMINFO_OUTPUT =~ "gfx94" ]]; then
                    MIOPEN_TEST_GFX94X=true
                fi
            else
                echo "TESTING IS NOT SUPPORTED FOR THE DETECTED GPU"
                MIOPEN_TEST_GPU_DETECTION_FAILED=true
            fi
        else
            echo "ROCMINFO NOT FOUND, GPU TYPE UNKNOWN. Manually set respective MIOPEN_TEST_GFX* CMake variable to specify target GPU for testing."
            MIOPEN_TEST_GPU_DETECTION_FAILED=true
        fi
    fi
}

# Call the gpu_detection function
gpu_detection

# Locate MIOpenDriver binary
MIOpenDriver_path="/MIOpen/build/bin/MIOpenDriver"  # Change this to the actual path

if [ "$(compare_env_variable "MIOPEN_TEST_WITH_MIOPENDRIVER" "true" "True" "on" "On" "ON")" = true ]; then
    if [ "$(compare_env_variable "MIOPEN_TEST_ALL" "1" "true" "True" "on" "On" "ON")" = true ]; then
        #test_miopendriver_regression_issue_1576, FLOAT_DISABLED HALF_ENABLED
        if [ "$MIOPEN_TEST_FLOAT_ARG" = "--half" ]; then    
            echo "Running test_miopendriver_regression_issue_1576"
            export MIOPEN_FIND_MODE=1
            export MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvDirectNaiveConvBwd
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV --forw 2 --in_layout NCHW --out_layout NCHW --fil_layout NCHW -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
            unset MIOPEN_FIND_MODE
            unset MIOPEN_DEBUG_FIND_ONLY_SOLVER
        fi

        #test_miopendriver_regression_half, FLOAT_DISABLED HALF_ENABLED
        if [ "$MIOPEN_TEST_FLOAT_ARG" = "--half" ]; then
            echo "Running test_miopendriver_regression_half"
            # WORKAROUND_ISSUE_2110_2: tests for 2110 and 2160 shall be added to "test_pooling3d --all" but this is
            # impossible until backward pooling limitation (issue #2110 (2)) is fully fixed.
            # Partial (3D only) regression test for https://github.com/ROCm/MIOpen/issues/2160.
            $MIOpenDriver_path $MIOPENDRIVER_MODE_POOL -M 0 --input 1x64x41x40x70 -y 41 -x 40 -Z 70 -m avg -F 1 -t 1 -i 1
            # Partial (3D only) regression test for https://github.com/ROCm/MIOpen/issues/2110 (1).
            $MIOpenDriver_path $MIOPENDRIVER_MODE_POOL -M 0 --input 1x64x41x40x100 -y 4 -x 4 -Z 100 -m max -F 1 -t 1 -i 1
        fi

        #test_miopendriver_regression_int8, FLOAT_DISABLED INT8_ENABLED
        if [ "$MIOPEN_TEST_FLOAT_ARG" = "--int8" ]; then    
            echo "Running test_miopendriver_regression_int8"
            export MIOPEN_FIND_MODE=1
            export MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvDirectNaiveConvFwd
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV --forw 2 --in_layout NCHW --out_layout NCHW --fil_layout NCHW -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
            unset MIOPEN_FIND_MODE
            unset MIOPEN_DEBUG_FIND_ONLY_SOLVER
        fi

        #test_miopendriver_regression_float_half_gfx10, HALF_ENABLED
        if [[ ( "$MIOPEN_TEST_FLOAT_ARG" = "--half" || "$MIOPEN_TEST_FLOAT_ARG" = "--float" ) && \
            "$MIOPEN_TEST_GFX103X" = true ]]; then
        
            echo "Running test_miopendriver_regression_float_half_gfx10"
            # Regression test for:
            #   [Navi21] Fixing Batchnorm backward precision issues by adjusting workgroup size (SWDEV-292187, SWDEV-319919)
            #   https://github.com/ROCm/MIOpen/pull/1386
            $MIOpenDriver_path $MIOPENDRIVER_MODE_BN -n 256 -c 512 -H 18 -W 18 -m 1 --forw 0 -b 1 -r 1
            $MIOpenDriver_path $MIOPENDRIVER_MODE_BN -n 256 -c 512 -H 28 -W 28 -m 1 --forw 0 -b 1 -r 1
        fi

        #test_miopendriver_regression_big_tensor 
        if [ "$MIOPEN_TEST_FLOAT_ARG" = "--float" ] && \
        { [ "$MIOPEN_TEST_GFX90A" = true ] || [ "$MIOPEN_TEST_GFX94X" = true ] || [ "$MIOPEN_TEST_GFX103X" = true ]; }; then
            echo "Running test_miopendriver_regression_big_tensor"
            # Regression test for https://github.com/ROCm/MIOpen/issues/1661
            # Issue #1697: this is large test which has to run in serial and not enabled on gfx900/gfx906
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -W 5078 -H 4903 -c 24 -n 5 -k 1 --fil_w 3 --fil_h 3 --pad_w 6 --pad_h 4 -F 1
        fi

        #test_miopendriver_regression_half_gfx9 
        if [ "$MIOPEN_TEST_FLOAT_ARG" = "--float" ] && \
        { [ "$MIOPEN_TEST_GFX90A" = true ] || [ "$MIOPEN_TEST_GFX94X" = true ]; }; then
            echo "Running test_miopendriver_regression_half_gfx9"
            # Regression test for:
            #   [SWDEV-375617] Fix 3d convolution Host API bug
            #   https://github.com/ROCm/MIOpen/pull/1935
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -n 2 -c 64 --in_d 128 -H 128 -W 128 -k 32 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1
        fi

        #test_miopendriver_conv2d_trans 
        if [[ ( "$MIOPEN_TEST_FLOAT_ARG" = "--half" || "$MIOPEN_TEST_FLOAT_ARG" = "--bf16" || "$MIOPEN_TEST_FLOAT_ARG" = "--float" ) && \
            "$MIOPEN_TEST_GFX900" != true ]]; then
            echo "Running test_miopendriver_conv2d_trans"
            # Why we have to use the driver:
            #   The transposed convolutions are paritally implemented in the convolution_api layer,
            #   but test apps (including test_conv*) were designed as unit tests and, therefore, do not use the public API.
            # Also serves as a regression test for https://github.com/ROCm/MIOpen/issues/2459.
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -m trans -x 1 -y 1 -W 112 -H 112 -c 64 -n 8 -k 32 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 0 -V 1
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -m trans -x 1 -y 7 -W 17 -H 17 -c 32 -n 128 -k 16 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -g 2 -F 0 -V 1
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -m trans -x 10 -y 5 -W 341 -H 79 -c 32 -n 4 -k 8 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 4 -F 0 -V 1
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -m trans -x 20 -y 5 -W 700 -H 161 -c 1 -n 4 -k 32 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -m trans -x 3 -y 3 -W 108 -H 108 -c 3 -n 8 -k 64 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -m trans -x 5 -y 5 -W 175 -H 40 -c 128 -n 16 -k 256 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -m trans -x 5 -y 5 -W 700 -H 161 -c 1 -n 16 -k 64 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -m trans -x 7 -y 7 -W 224 -H 224 -c 3 -n 16 -k 64 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1
        fi
    else #"$MIOPEN_TEST_ALL" != true
        #test_miopendriver_regression_issue_2047
        if [[ ( "$MIOPEN_TEST_FLOAT_ARG" = "--half" || "$MIOPEN_TEST_FLOAT_ARG" = "--bf16" || "$MIOPEN_TEST_FLOAT_ARG" = "--float" ) && \
            "$MIOPEN_TEST_GFX900" != true ]]; then
            echo "Running test_miopendriver_regression_issue_2047"
            # Regression test for: MIOpenIm3d2Col stuck with ROCm update, https://github.com/ROCm/MIOpen/issues/2047
            export MIOPEN_FIND_MODE=normal
            export MIOPEN_DEBUG_FIND_ONLY_SOLVER=GemmFwdRest
            $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV -n 1 -c 1 --in_d 2 -H 1 -W 2 -k 2 --fil_d 2 -y 1 -x 2 \
                                                        --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 \
                                                        --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -i 1 -t 1 -w 1
            unset MIOPEN_FIND_MODE
            unset MIOPEN_DEBUG_FIND_ONLY_SOLVER
        fi

        #smoke_miopendriver_gemm   
        if [ "$MIOPEN_TEST_FLOAT_ARG" = "--half" ] || [ "$MIOPEN_TEST_FLOAT_ARG" = "--float" ]; then
            echo "Running smoke_miopendriver_gemm "
            $MIOpenDriver_path $MIOPENDRIVER_MODE_GEMM -m 256 -n 512 -k 1024 -i 1 -V 1
        fi
    fi
fi

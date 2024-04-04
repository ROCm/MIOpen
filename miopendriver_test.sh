#!/bin/bash

# Define default values
MIOPEN_DEFAULT_BACKEND="HIP"
MIOPEN_BACKEND="$MIOPEN_DEFAULT_BACKEND"
MIOPEN_EMBED_BUILD="Off"  # Default is Off
MIOPEN_TEST_ALL="OFF"  # Assuming MIOPEN_TEST_ALL is initially set to OFF

# Check if MIOPEN_BACKEND is overwritten by external flag
if [ -n "$1" ]; then
    MIOPEN_BACKEND="$1"
fi

# Check if MIOPEN_EMBED_BUILD is overwritten by external flag
if [ -n "$2" ]; then
    MIOPEN_EMBED_BUILD="$2"
fi

# Check if MIOPEN_TEST_ALL is overwritten by external flag
if [ -n "$3" ]; then
    MIOPEN_TEST_ALL="$3"
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

# Determine MIOPENDRIVER_MODE_CONV based on options
if [ "$MIOPEN_TEST_HALF" = "ON" ]; then
    MIOPEN_TEST_FLOAT_ARG=--half
    MIOPENDRIVER_MODE_CONV=convfp16
elif [ "$MIOPEN_TEST_INT8" = "ON" ]; then
    MIOPEN_TEST_FLOAT_ARG=--int8
    MIOPENDRIVER_MODE_CONV=convint8
elif [ "$MIOPEN_TEST_BFLOAT16" = "ON" ]; then
    MIOPEN_TEST_FLOAT_ARG=--bfloat16
    MIOPENDRIVER_MODE_CONV=convbfp16
else
    MIOPEN_TEST_FLOAT_ARG=--float
    MIOPENDRIVER_MODE_CONV=conv
fi

# Execute MIOpenDriver binary
MIOpenDriver_path="/MIOpen/build/bin/MIOpenDriver"  # Change this to the actual path

if [ "$MIOPEN_TEST_WITH_MIOPENDRIVER" = true ]; then
    if [ "$MIOPEN_TEST_FLOAT_ARG" = "--half" ]; then
        #test_miopendriver_regression_half, FLOAT_DISABLED HALF_ENABLED
        echo "Running test_miopendriver_regression_half"
        $MIOpenDriver_path $MIOPENDRIVER_MODE_CONV --forw 2 --in_layout NCHW --out_layout NCHW --fil_layout NCHW -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
    fi
fi

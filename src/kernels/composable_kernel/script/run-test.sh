 #!/bin/bash
 unset MIOPEN_DEBUG_CONV_WINOGRAD
 unset MIOPEN_DEBUG_CONV_FFT
 unset MIOPEN_DEBUG_CONV_DIRECT
 unset MIOPEN_DEBUG_CONV_GEMM
 unset MIOPEN_DEBUG_CONV_SCGEMM
 unset MIOPEN_DEBUG_CONV_IMPLICIT_GEMM
 unset MIOPEN_DEBUG_FIND_ONLY_SOLVER
 unset MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_FIRST_SOLUTION
 unset MIOPEN_FIND_ENFORCE
 unset MIOPEN_LOG_LEVEL

 export MIOPEN_DEBUG_CONV_WINOGRAD=0
 export MIOPEN_DEBUG_CONV_FFT=0
 export MIOPEN_DEBUG_CONV_DIRECT=0
 export MIOPEN_DEBUG_CONV_GEMM=0
 export MIOPEN_DEBUG_CONV_SCGEMM=0
 export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1

## implicit GEMM
 export MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS=1
 export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS=0
 export MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM=0
 export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE=0

## GPU
#export ROCR_VISIBLE_DEVICES=0
#export HIP_VISIBLE_DEVICES=0
#export GPU_DEVICE_ORDINAL=0

## debug
#export HSA_TOOLS_LIB=librocr_debug_agent64.so

## find_db
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
## enforce searching
# export MIOPEN_FIND_ENFORCE=4
## log
 export MIOPEN_LOG_LEVEL=6
 export MIOPEN_LOG_LEVEL=0
## rocblas
 export ROCBLAS_LAYER=3

## HIP
 export KMDUMPISA=1
 export KMDUMPLLVM=1
 export KMDUMPDIR=$PWD

## db path
export MIOPEN_USER_DB_PATH=./db_config/

make -j MIOpenDriver
#make -j test_conv2d

export MIOPEN_DEBUG_FIND_ONLY_SOLVER=31

DIR=4
./bin/MIOpenDriver conv -F $DIR -n 128 -c 128 -H 32 -W 32 -k 128 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1 -w 1
./bin/MIOpenDriver conv -F $DIR -n 64 -c 1024 -H 56 -W 56 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -w 1
./bin/MIOpenDriver conv -F $DIR -n 64 -c 128 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -w 1
./bin/MIOpenDriver conv -F $DIR -n 64 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -w 1
./bin/MIOpenDriver conv -F $DIR -n 128 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -w 1
./bin/MIOpenDriver conv -F $DIR -n 128 -c 128 -H 16 -W 16 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -w 1
echo "end"

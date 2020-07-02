#!/bin/bash

# syntax:
# miopen_gridwise_gemm_builder input_file output_file isa_version (compile-time flags)

# enable bash debugging
KMDBSCRIPT="${KMDBSCRIPT:=0}"
if [ $KMDBSCRIPT == "1" ]
then
  set -x
fi

# pass extra options to OPT
# KMOPTOPT can be used to pass last-minute options to opt in the backend
# if not set, then "-O3" would be passed to opt
KMOPTOPT="${KMOPTOPT:="-O3"}"

# pass extra options to LLC
# KMOPTLLC can be used to pass last-minute options to llc in the backend
# if not set, then "-O2" will be passed to llc
KMOPTLLC="${KMOPTLLC:="-O3"}"

# prepare env vars.
COMPILER_DIR=/opt/rocm/llvm
BIN_DIR=$COMPILER_DIR/bin
ROCM_LIB=/opt/rocm/lib

CLANG=$BIN_DIR/clang-11
LLVM_LINK=$BIN_DIR/llvm-link
OPT=$BIN_DIR/opt
LLC=$BIN_DIR/llc
LLD=$BIN_DIR/lld
BUNDLER=$BIN_DIR/clang-offload-bundler

# check command line arguments.
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 input_file output_file isa_version (compile-time flags)" >&2
  exit 1
fi

# parse input.
INPUT_FILE=$1
shift
OUTPUT_FILE=$1
shift
AMDGPU_TARGET=$1
shift
TMP_DIR=$1
shift

echo "gpu arch: $AMDGPU_TARGET"

if [ $AMDGPU_TARGET == "gfx906" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_906.amdgcn.bc"
  #KMOPTLLC+=" -mattr=+sram-ecc"
elif [ $AMDGPU_TARGET == "gfx908" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_908.amdgcn.bc"
  #KMOPTLLC+=" -mattr=+sram-ecc"
fi

# launch the frontend.
$CLANG  -cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -emit-llvm-bc \
-emit-llvm-uselists -disable-free -disable-llvm-verifier -discard-value-names \
-main-file-name gridwise_convolution_implicit_gemm_v4r1_wrw_nchw_kcyx_nkhw_lds_double_buffer.cpp \
-mrelocation-model pic -pic-level 1 -mthread-model posix -mframe-pointer=none -fno-rounding-math \
-mconstructor-aliases -aux-target-cpu x86-64 -target-cpu $AMDGPU_TARGET -fcuda-is-device \
-fcuda-allow-variadic-functions -fvisibility hidden -fapply-global-visibility-to-externs \
-mlink-builtin-bitcode /opt/rocm/lib/hip.amdgcn.bc -mlink-builtin-bitcode /opt/rocm/lib/ocml.amdgcn.bc \
-mlink-builtin-bitcode /opt/rocm/lib/ockl.amdgcn.bc \
-mlink-builtin-bitcode /opt/rocm/lib/oclc_finite_only_off.amdgcn.bc \
-mlink-builtin-bitcode /opt/rocm/lib/oclc_daz_opt_off.amdgcn.bc \
-mlink-builtin-bitcode /opt/rocm/lib/oclc_correctly_rounded_sqrt_on.amdgcn.bc \
-mlink-builtin-bitcode /opt/rocm/lib/oclc_unsafe_math_off.amdgcn.bc \
-mlink-builtin-bitcode /opt/rocm/lib/oclc_isa_version_906.amdgcn.bc \
-mlink-builtin-bitcode /opt/rocm/lib/oclc_wavefrontsize64_on.amdgcn.bc \
-target-cpu $AMDGPU_TARGET -dwarf-column-info -fno-split-dwarf-inlining -debugger-tuning=gdb -v \
-resource-dir /opt/rocm-3.5.0/llvm/lib/clang/11.0.0 -isystem /opt/rocm-3.5.0/hip/../include \
-isystem /opt/rocm/llvm/lib/clang/11.0.0/include/.. -isystem /opt/rocm-3.5.0/hip/include \
-isystem /opt/rocm/include -I . -D __HIP_ROCclr__=1 -D __HIP_PLATFORM_HCC__=1 -D __HIP_ROCclr__=1 \
 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 \
 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 \
 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 \
 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward \
 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 \
 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 \
 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 \
 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward \
 -internal-isystem /usr/local/include -internal-isystem /opt/rocm-3.5.0/llvm/lib/clang/11.0.0/include \
 -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include \
 -internal-externc-isystem /usr/include -internal-isystem /usr/local/include \
 -internal-isystem /opt/rocm-3.5.0/llvm/lib/clang/11.0.0/include \
 -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include \
 -internal-externc-isystem /usr/include -O3 -Werror -Weverything -Wno-c++98-compat \
 -Wno-c++98-compat-pedantic -Wno-conversion -Wno-double-promotion -Wno-exit-time-destructors \
 -Wno-extra-semi -Wno-float-conversion -Wno-gnu-anonymous-struct \
 -Wno-gnu-zero-variadic-macro-arguments -Wno-missing-prototypes -Wno-nested-anon-types \
 -Wno-padded -Wno-return-std-move-in-c++11 -Wno-shorten-64-to-32 -Wno-sign-conversion \
 -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-weak-vtables \
 -Wno-covered-switch-default -Wno-disabled-macro-expansion -Wno-undefined-reinterpret-cast \
 -Wno-unused-command-line-argument -std=c++14 -fdeprecated-macro -fno-autolink \
 -fdebug-compilation-dir $TMP_DIR \
 -ferror-limit 19 -fgnuc-version=4.2.1 -fcxx-exceptions -fexceptions -fcolor-diagnostics \
 -vectorize-loops -vectorize-slp -mllvm -amdgpu-early-inline-all=true -mllvm \
 -amdgpu-function-calls=false -mllvm -amdgpu-enable-global-sgpr-addr \
 -mllvm --amdgpu-spill-vgpr-to-agpr=0 -fcuda-allow-variadic-functions -faddrsig -o \
 $INPUT_FILE.bc -x hip $INPUT_FILE $* \
 -Wno-unused-macros -Wno-error \
 -include-pch /opt/rocm/hip/include/hip/hip_runtime.h.pch

$LLVM_LINK $INPUT_FILE.bc -o $INPUT_FILE-linked.bc

# optimize bitcodes.
$OPT $INPUT_FILE-linked.bc $KMOPTOPT -mtriple=amdgcn-amd-amdhsa -mcpu=$AMDGPU_TARGET \
-amdgpu-early-inline-all=true -amdgpu-function-calls=false -amdgpu-enable-global-sgpr-addr \
--amdgpu-spill-vgpr-to-agpr=0 -o $INPUT_FILE-optimized.bc

# launch llc to produce ISA.
$LLC $INPUT_FILE-optimized.bc $KMOPTLLC -mtriple=amdgcn-amd-amdhsa -mcpu=$AMDGPU_TARGET \
-filetype=obj -amdgpu-early-inline-all=true -amdgpu-function-calls=false \
-amdgpu-enable-global-sgpr-addr --amdgpu-spill-vgpr-to-agpr=0 -o $INPUT_FILE-out.o

# launch lld 
$LLD -flavor gnu --no-undefined -shared -o $INPUT_FILE.out $INPUT_FILE-out.o

# bundler
$BUNDLER -type=o -targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-$AMDGPU_TARGET \
-inputs=/dev/null,$INPUT_FILE.out \
-outputs=$OUTPUT_FILE
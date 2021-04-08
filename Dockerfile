FROM ubuntu:18.04

ARG PREFIX=/usr/local
ARG GPU_ARCH=";"
ARG MIOTENSILE_VER="default"
ARG USE_TARGETID="OFF"

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
RUN if [ "$USE_TARGETID" = "ON" ] ; then sh -c 'echo deb [arch=amd64 trusted=yes] http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/ compute-rocm-dkms-no-npi-hipclang 6416 > /etc/apt/sources.list.d/rocm.list'; else sh -c 'echo deb [arch=amd64 trusted=yes] http://repo.radeon.com/rocm/apt/.apt_3.7/ xenial main > /etc/apt/sources.list.d/rocm.list'; fi
RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu xenial main universe | tee -a /etc/apt/sources.list"

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    cmake \
    comgr \
    curl \
    clang-format-3.8 \
    clang-3.8 \
    clang-tidy-3.8\
    doxygen \
    g++ \
    gdb \
    git \
    hip-rocclr \
    lcov \
    libelf-dev \
    libncurses5-dev \
    libnuma-dev \
    libpthread-stubs0-dev \
    llvm-amdgpu \
    miopengemm \
    pkg-config \
    python \
    python3 \
    python-dev \
    python3-dev \
    python-pip \
    python3-pip \
    python3-distutils \
    python3-venv \
    software-properties-common \
    wget \
    rocm-dev \
    rocm-device-libs \
    rocm-opencl \
    rocm-opencl-dev \
    rocm-cmake \
    rocblas \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb
RUN dpkg -i dumb-init_*.deb && rm dumb-init_*.deb

# Install cget
RUN pip3 install cget

# Add symlink to /opt/rocm
RUN [ -d /opt/rocm ] || ln -sd $(realpath /opt/rocm-*) /opt/rocm

# Make sure /opt/rcom is in the paths
ENV PATH="/opt/rocm:${PATH}"

# Build using hip-clang
RUN cget -p $PREFIX init --cxx /opt/rocm/llvm/bin/clang++ --std=c++14 -DAMDGPU_TARGETS=${GPU_ARCH}

# Install dependencies
RUN cget -p $PREFIX install pfultz2/rocm-recipes
# Install a newer version of cmake for libMLIRMIOpen
RUN cget -p $PREFIX install kitware/cmake@v3.13.4
ADD min-requirements.txt /min-requirements.txt
RUN CXXFLAGS='-isystem $PREFIX/include' cget -p $PREFIX install -f /min-requirements.txt
RUN cget -p $PREFIX install danmar/cppcheck@dd05839a7e63ef04afd34711cb3e1e0ef742882f

# Install doc requirements
ADD doc/requirements.txt /doc-requirements.txt
RUN pip install -r /doc-requirements.txt

# Use parallel job to accelerate tensile build
# Workaround for Tensile with TargetID feature
RUN if [ "$USE_TARGETID" = "ON" ] ; then export HIPCC_LINK_FLAGS_APPEND='-O3 -parallel-jobs=4' && export HIPCC_COMPILE_FLAGS_APPEND='-O3 -Wno-format-nonliteral -parallel-jobs=4' && rm /usr/bin/hipcc; fi

# install last released miopentensile in default (master), install latest commits when MIOTENSILE_VER="latest" (develop)
RUN if [ "$USE_TARGETID" = "OFF" ] ; then echo "MIOpenTensile is not installed."; elif [ "$MIOTENSILE_VER" = "latest" ] ; then cget -p $PREFIX install ROCmSoftwarePlatform/MIOpenTensile@be26d30d3d7509a414134a45f4a6d49e5da250b8; else cget -p $PREFIX install ROCmSoftwarePlatform/MIOpenTensile@4bfe00a8de61d12862d9fa803b8ea9a981a50f97; fi

RUN cd ~ && \
    export MLIR_COMMIT=31d92f4c64ae6fa6b7c5d543f68b69300b4513ce && \
    wget https://github.com/ROCmSoftwarePlatform/llvm-project-mlir/archive/$MLIR_COMMIT.tar.gz && \
    tar -xvzf $MLIR_COMMIT.tar.gz && \
    rm -rf $MLIR_COMMIT.tar.gz && \
    cd llvm-project-mlir-$MLIR_COMMIT && mkdir -p build && cd build && \
    $PREFIX/bin/cmake -G "Unix Makefiles" ../llvm \
      -DLLVM_ENABLE_PROJECTS="mlir;lld" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLVM_BUILD_LLVM_DYLIB=OFF \
      -DLLVM_ENABLE_TERMINFO=OFF && \
    make -j$(nproc) libMLIRMIOpen && \
    cd ~ && rm -rf llvm-project-mlir-$MLIR_COMMIT

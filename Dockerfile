FROM ubuntu:18.04

ARG PREFIX=/usr/local
ARG GPU_ARCH=";"
ARG MIOTENSILE_VER="default"
ARG USE_TARGETID="OFF"
ARG USE_MLIR="OFF"
ARG USE_FIN="OFF"

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
# Note: The ROCm version with $USE_MLIR should keep in sync with default ROCm version
# unless MLIR library is incompatible with current ROCm.

RUN if [ "$USE_TARGETID" = "ON" ] ; \
        then export ROCM_APT_VER=.apt_4.1.1;\
    elif [ "$USE_MLIR" = "ON" ] ; \
        then export ROCM_APT_VER=.apt_4.2;\
    else export ROCM_APT_VER=.apt_4.2;  \
    fi && \
echo $ROCM_APT_VER &&\
sh -c 'echo deb [arch=amd64 trusted=yes] http://compute-artifactory.amd.com/artifactory/list/rocm-release-archive-deb/ 4.3 rel-34 > /etc/apt/sources.list.d/rocm.list'
RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu xenial main universe | tee -a /etc/apt/sources.list"

#Add gpg keys
# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    wget \
    ca-certificates \
    curl \
    libnuma-dev \
    gnupg && \
#get -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \ THE gpg key doesn't work on bigred2 nodes
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    cmake \
    comgr \
    clang-format-10 \
    doxygen \
    g++ \
    gdb \
    git \
    hip-rocclr \
    lcov \
    libelf-dev \
    libncurses5-dev \
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
    rocm-dev \
    rocm-device-libs \
    rocm-opencl \
    rocm-opencl-dev \
    rocm-cmake \
    rocblas \
    zlib1g-dev \
    kmod && \
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
RUN cget -p $PREFIX install kitware/cmake@v3.15.1

ADD min-requirements.txt /min-requirements.txt
RUN CXXFLAGS='-isystem $PREFIX/include' cget -p $PREFIX install -f /min-requirements.txt
RUN if [ "$USE_FIN" = "ON" ]; then cget -p $PREFIX install nlohmann/json@350ff4f7ced7c4117eae2fb93df02823c8021fcb; fi
RUN cget -p $PREFIX install danmar/cppcheck@dd05839a7e63ef04afd34711cb3e1e0ef742882f

# Install doc requirements
ADD doc/requirements.txt /doc-requirements.txt
RUN pip install -r /doc-requirements.txt

# Use parallel job to accelerate tensile build
# Workaround for Tensile with TargetID feature
RUN if [ "$USE_TARGETID" = "ON" ] ; then export HIPCC_LINK_FLAGS_APPEND='-O3 -parallel-jobs=4' && export HIPCC_COMPILE_FLAGS_APPEND='-O3 -Wno-format-nonliteral -parallel-jobs=4' && rm /usr/bin/hipcc; fi

# install last released miopentensile in default (master), install latest commits when MIOTENSILE_VER="latest" (develop)
RUN if [ "$USE_TARGETID" = "OFF" ] ; then echo "MIOpenTensile is not installed."; elif [ "$MIOTENSILE_VER" = "latest" ] ; then cget -p $PREFIX install ROCmSoftwarePlatform/MIOpenTensile@94a9047741d16a8eccd290131b78fb1aa69cdcdf; else cget -p $PREFIX install ROCmSoftwarePlatform/MIOpenTensile@94a9047741d16a8eccd290131b78fb1aa69cdcdf; fi

RUN if [ "$USE_MLIR" = "ON" ]; \
    then cd ~ && \
    export MLIR_COMMIT=7416cfaee140068921b64996ba945ce615c36f44 && \
    wget https://github.com/ROCmSoftwarePlatform/llvm-project-mlir/archive/$MLIR_COMMIT.tar.gz && \
    tar -xvzf $MLIR_COMMIT.tar.gz && \
    rm -rf $MLIR_COMMIT.tar.gz && \
    cd llvm-project-mlir-$MLIR_COMMIT && mkdir -p build && cd build && \
    $PREFIX/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FAT_LIBMLIRMIOPEN=1 && \
    make -j$(nproc) libMLIRMIOpen && \
    $PREFIX/bin/cmake --install . --component libMLIRMIOpen --prefix /opt/rocm && \
    cd ~ && rm -rf llvm-project-mlir-$MLIR_COMMIT; fi

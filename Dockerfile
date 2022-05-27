FROM ubuntu:18.04 as composable_kernel
ARG ROCMVERSION=5.1

RUN set -xe

ARG BUILD_THREADS=8
ARG DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/.apt_$ROCMVERSION/
# Add rocm repository
RUN apt-get update
RUN apt-get install -y wget gnupg
RUN wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN sh -c "echo deb [arch=amd64] $DEB_ROCM_REPO ubuntu main > /etc/apt/sources.list.d/rocm.list"
RUN wget --no-check-certificate -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN sh -c "echo deb https://apt.kitware.com/ubuntu/ bionic main | tee -a /etc/apt/sources.list"

# ADD requirements.txt requirements.txt
# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    sshpass \
    build-essential \
    cmake-data=3.15.1-0kitware1 \
    cmake=3.15.1-0kitware1 \
    curl \
    doxygen \
    g++ \
    git \
    hip-rocclr \
    jq \
    libelf-dev \
    libncurses5-dev \
    libnuma-dev \
    libpthread-stubs0-dev \
    llvm-amdgpu \
    pkg-config \
    python \
    python3 \
    python-dev \
    python3-dev \
    python-pip \
    python3-pip \
    software-properties-common \
    wget \
    rocm-dev \
    rocm-device-libs \
    rocm-cmake \
    vim \
    zlib1g-dev \
    openssh-server \
    kmod && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Setup ubsan environment to printstacktrace
RUN ln -s /usr/bin/llvm-symbolizer-3.8 /usr/local/bin/llvm-symbolizer
ENV UBSAN_OPTIONS=print_stacktrace=1

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb
RUN dpkg -i dumb-init_*.deb && rm dumb-init_*.deb

# Install cget
RUN pip install cget

# Install rclone
RUN pip install https://github.com/pfultz2/rclone/archive/master.tar.gz

ARG PREFIX=/opt/rocm
# Install dependencies
RUN cget install pfultz2/rocm-recipes
# Install rbuild
RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/6d78a0553babdaea8d2da5de15cbda7e869594b8.tar.gz
# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN cget -p $PREFIX install ROCmSoftwarePlatform/rocm-recipes
RUN groupadd -f render
WORKDIR /root
RUN rm -rf /tmp/ccache* && mkdir /tmp/ccache && wget https://github.com/ccache/ccache/archive/7f1572ae9ca958fa923a66235f6a64a360b03523.tar.gz -O /tmp/ccache.tar.gz && \
    tar zxvf /tmp/ccache.tar.gz -C /tmp/ && mkdir /tmp/ccache-7f1572ae9ca958fa923a66235f6a64a360b03523/build && \
    cd /tmp/ccache-7f1572ae9ca958fa923a66235f6a64a360b03523/build && \
    cmake -DZSTD_FROM_INTERNET=ON -DHIREDIS_FROM_INTERNET=ON .. && make -j install
RUN ccache -s 
ARG LLVM_COMMIT=5cba46feb6af367b1cafaa183ec42dbfb8207b14
ARG CCACHE_SECONDARY_STORAGE=""
ARG CCACHE_DIR="/tmp"
ARG COMPILER_LAUNCHER=""
RUN wget -O llvm.tar.gz https://github.com/RadeonOpenCompute/llvm-project/archive/${LLVM_COMMIT}.tar.gz && \
    tar zxvf llvm.tar.gz && \
    cd llvm-project-${LLVM_COMMIT} && \
    mkdir build && cd build &&\
    cmake -DCMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" -DCMAKE_INSTALL_PREFIX=/opt/rocm-${ROCMVERSION}.0/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" ../llvm && \
    make -j $(nproc) install 
ARG CK_COMMIT=91d8b7d67ae9dbf8a6e691ea3e17c0b9705c6ba7
RUN  wget -O ck.tar.gz https://www.github.com/rocmsoftwareplatform/composable_kernel/archive/${CK_COMMIT}.tar.gz && \
    tar zxvf ck.tar.gz &&\
    cd composable_kernel-${CK_COMMIT} && \
    mkdir build && cd build && \
    CXX=/opt/rocm-${ROCMVERSION}.0/bin/hipcc cmake -DCMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" -DCMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_FLAGS=" --offload-arch=gfx900 --offload-arch=gfx906 --offload-arch=gfx908 --offload-arch=gfx90a -O3 " .. && \
    make -j $(nproc) package && cp *.deb /root/


FROM ubuntu:18.04 as miopen

ARG USE_MLIR="OFF"

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
# Note: The ROCm version with $USE_MLIR should keep in sync with default ROCm version
# unless MLIR library is incompatible with current ROCm.
RUN apt-get update
RUN apt-get install -y wget gnupg
RUN if [ "$USE_MLIR" = "ON" ] ; \
        then export ROCM_APT_VER=.apt_5.1;\
    else \
        export ROCM_APT_VER=.apt_5.1;  \
    fi && \
echo $ROCM_APT_VER &&\
sh -c 'echo deb [arch=amd64 trusted=yes] http://repo.radeon.com/rocm/apt/$ROCM_APT_VER/ ubuntu main > /etc/apt/sources.list.d/rocm.list'
RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu bionic main universe | tee -a /etc/apt/sources.list"
RUN wget --no-check-certificate -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN sh -c "echo deb https://apt.kitware.com/ubuntu/ bionic main | tee -a /etc/apt/sources.list"

#Add gpg keys
# Install dependencies
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    wget \
    ca-certificates \
    curl \
    libnuma-dev \
    gnupg \
    apt-utils && \
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 9386B48A1A693C5C && \
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    build-essential \
    cmake-data=3.15.1-0kitware1 \
    cmake=3.15.1-0kitware1 \
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
    rocblas \
    zlib1g-dev \
    kmod && \
    apt-get remove -y rocm-cmake && \
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
RUN pip3 install https://github.com/pfultz2/cget/archive/a426e4e5147d87ea421a3101e6a3beca541c8df8.tar.gz

# Install rbuild
RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/6d78a0553babdaea8d2da5de15cbda7e869594b8.tar.gz

# Add symlink to /opt/rocm
RUN [ -d /opt/rocm ] || ln -sd $(realpath /opt/rocm-*) /opt/rocm

# Make sure /opt/rcom is in the paths
ENV PATH="/opt/rocm:${PATH}"

# Add requirements files
ADD rbuild.ini /rbuild.ini
ADD requirements.txt /requirements.txt
ADD dev-requirements.txt /dev-requirements.txt
# Install dependencies
# TODO: Add --std=c++14
ARG GPU_ARCH=";"
ARG PREFIX=/usr/local
ARG USE_FIN="OFF"
ARG CCACHE_SECONDARY_STORAGE=""
ARG CCACHE_DIR="/tmp"
RUN env
# RUN cget -p $PREFIX install https://github.com/ccache/ccache/archive/7f1572ae9ca958fa923a66235f6a64a360b03523.tar.gz -DZSTD_FROM_INTERNET=ON -DHIREDIS_FROM_INTERNET=ON
RUN rm -rf /tmp/ccache* && mkdir /tmp/ccache && wget https://github.com/ccache/ccache/archive/7f1572ae9ca958fa923a66235f6a64a360b03523.tar.gz -O /tmp/ccache.tar.gz && \
    tar zxvf /tmp/ccache.tar.gz -C /tmp/ && mkdir /tmp/ccache-7f1572ae9ca958fa923a66235f6a64a360b03523/build && \
    cd /tmp/ccache-7f1572ae9ca958fa923a66235f6a64a360b03523/build && \
    cmake -DZSTD_FROM_INTERNET=ON -DHIREDIS_FROM_INTERNET=ON .. && make -j install
RUN ccache -s 
ARG COMPILER_LAUNCHER=""
RUN if [ "$USE_FIN" = "ON" ]; then \
        rbuild prepare -s fin -d $PREFIX -DAMDGPU_TARGETS=${GPU_ARCH} -DCMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}"; \
    else \
        rbuild prepare -s develop -d $PREFIX -DAMDGPU_TARGETS=${GPU_ARCH} -DCMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}"; \
    fi

RUN ccache -s 
# Install doc requirements
ADD doc/requirements.txt /doc-requirements.txt
RUN pip3 install -r /doc-requirements.txt

# Use parallel job to accelerate tensile build
# Workaround for Tensile with TargetID feature
ARG USE_TARGETID="OFF"
RUN if [ "$USE_TARGETID" = "ON" ] ; then export HIPCC_LINK_FLAGS_APPEND='-O3 -parallel-jobs=4' && export HIPCC_COMPILE_FLAGS_APPEND='-O3 -Wno-format-nonliteral -parallel-jobs=4' && rm -f /usr/bin/hipcc; fi

# install last released miopentensile in default (master), install latest commits when MIOTENSILE_VER="latest" (develop)
ARG MIOTENSILE_VER="default"
RUN if [ "$USE_TARGETID" = "OFF" ] ; then echo "MIOpenTensile is not installed."; elif [ "$MIOTENSILE_VER" = "latest" ] ; then cget -p $PREFIX install ROCmSoftwarePlatform/MIOpenTensile@94a9047741d16a8eccd290131b78fb1aa69cdcdf; else cget -p $PREFIX install ROCmSoftwarePlatform/MIOpenTensile@94a9047741d16a8eccd290131b78fb1aa69cdcdf; fi

RUN groupadd -f render

COPY --from=composable_kernel /root/composablekernel_*.deb /tmp/
RUN dpkg -i --force-all /tmp/composablekernel_*.deb

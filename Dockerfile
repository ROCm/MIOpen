FROM ubuntu:22.04 as miopen
ARG DEBIAN_FRONTEND=noninteractive

# Support multiarch
RUN dpkg --add-architecture i386

# Install preliminary dependencies
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    ca-certificates \
    curl \
    libnuma-dev \
    gnupg2 \
    wget

# This stage fails quite often and not always necessary
# RUN apt-get update && \
#     DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
#     "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"

#Add gpg keys
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn
RUN curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm-keyring.gpg

RUN wget https://repo.radeon.com/amdgpu-install/6.2.2/ubuntu/jammy/amdgpu-install_6.2.60202-1_all.deb --no-check-certificate
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    ./amdgpu-install_6.2.60202-1_all.deb

# Add rocm repository
RUN export ROCM_APT_VER=6.2.2;\
echo $ROCM_APT_VER &&\
sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/amdgpu/$ROCM_APT_VER/ubuntu jammy main > /etc/apt/sources.list.d/amdgpu.list' &&\
sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/$ROCM_APT_VER jammy main > /etc/apt/sources.list.d/rocm.list'
RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu jammy main universe | tee -a /etc/apt/sources.list"

RUN amdgpu-install -y --usecase=rocm --no-dkms

# Install dependencies
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    build-essential \
    cmake \
    clang-format-12 \
    doxygen \
    gdb \
    git \
    git-lfs \
    lbzip2 \
    lcov \
    libncurses5-dev \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    rocm-developer-tools \
    rocm-llvm-dev \
    rpm \
    software-properties-common && \
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
# GPU_ARCH can be defined in docker build process
ARG GPU_ARCHS=gfx908;gfx90a;gfx942;gfx1100
# install to /opt/rocm will cause permission issue
ARG PREFIX=/usr/local
ARG USE_FIN="OFF"
ARG CCACHE_SECONDARY_STORAGE=""
ARG CCACHE_DIR="/tmp"
RUN env
# RUN cget -p $PREFIX install https://github.com/ccache/ccache/archive/7f1572ae9ca958fa923a66235f6a64a360b03523.tar.gz -DZSTD_FROM_INTERNET=ON -DHIREDIS_FROM_INTERNET=ON
ARG CCACHE_COMMIT=7f1572ae9ca958fa923a66235f6a64a360b03523
RUN rm -rf /tmp/ccache* && mkdir /tmp/ccache
ADD https://github.com/ccache/ccache/archive/${CCACHE_COMMIT}.tar.gz /tmp/ccache.tar.gz
RUN tar zxvf /tmp/ccache.tar.gz -C /tmp/ && mkdir /tmp/ccache-${CCACHE_COMMIT}/build && \
    cd /tmp/ccache-${CCACHE_COMMIT}/build && \
    cmake -DZSTD_FROM_INTERNET=ON -DHIREDIS_FROM_INTERNET=ON .. && make -j install && rm -rf /tmp/*
RUN ccache -s 

# purge existing composable kernel installed with ROCm
# hence cannot use autoremove since it will remove more components
# even purge will remove some other components which is not ideal
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get purge -y --allow-unauthenticated \
    composablekernel-dev \
    miopen-hip

# TODO: it should be able to automatically get commit hash from requirements.txt
ARG CK_COMMIT=fb948120d2d674607e70d0e7587dcb249c0e74c7
RUN wget -O ck.tar.gz https://www.github.com/ROCm/composable_kernel/archive/${CK_COMMIT}.tar.gz && \
    tar zxvf ck.tar.gz &&\
    cd composable_kernel-${CK_COMMIT} && \
    mkdir build && cd build && \
    CXX=/opt/rocm/bin/amdclang++ cmake \
    -D CMAKE_PREFIX_PATH=/opt/rocm \
    -D CMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D GPU_ARCHS="gfx908;gfx90a;gfx942;gfx1100" \
    -D CMAKE_CXX_FLAGS=" -O3 " .. && \
    make -j $(nproc) install 

# Composable Kernel installed separated from rbuild to take in values from GPU_ARCHS 
# this can minimize build time
RUN sed -i '/composable_kernel/d' /requirements.txt

ARG COMPILER_LAUNCHER=""
# rbuild is used to trigger build of requirements.txt, dev-requirements.txt
RUN if [ "$USE_FIN" = "ON" ]; then \
        rbuild prepare -s fin -d $PREFIX -DGPU_ARCHS="${GPU_ARCHS}" -DCMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}"; \
    else \
        rbuild prepare -s develop -d $PREFIX -DGPU_ARCHS="${GPU_ARCHS}" -DCMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}"; \
    fi

RUN ccache -s 
# Install doc requirements
ADD docs/sphinx/requirements.txt /doc-requirements.txt
RUN pip3 install -r /doc-requirements.txt

# Composable Kernel requires this version cmake
RUN pip3 install --upgrade cmake==3.27.5

# groupadd can add one group a time
RUN groupadd -f render
RUN groupadd -f video
RUN usermod -a -G render,video root
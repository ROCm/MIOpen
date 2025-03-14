FROM ubuntu:22.04 as miopen
ARG DEBIAN_FRONTEND=noninteractive
# install to /opt/rocm will cause permission issue
ARG PREFIX=/usr/local
ARG USE_FIN="OFF"

ARG CCACHE_SECONDARY_STORAGE=""
ARG CCACHE_DIR="/tmp"
ARG CCACHE_COMMIT=7f1572ae9ca958fa923a66235f6a64a360b03523

# GPU_ARCHS should be defined as a build arg rather than hardcoded here. 
ARG GPU_ARCHS=none

ARG INSTALL_MIOPEN=OFF
ARG FRECKLE=0
ARG COMPILER_LAUNCHER=""
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn

# Support multiarch
RUN dpkg --add-architecture i386

# Install preliminary dependencies and add rocm gpg key
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
        apt-utils ca-certificates curl libnuma-dev gnupg2 wget  && \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm-keyring.gpg

# Get and install amdgpu-install.
RUN wget https://repo.radeon.com/amdgpu-install/6.3.2/ubuntu/jammy/amdgpu-install_6.3.60302-1_all.deb --no-check-certificate && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
       ./amdgpu-install_6.3.60302-1_all.deb

# Add rocm repository
RUN export ROCM_APT_VER=6.3.2; \
    echo $ROCM_APT_VER &&\
    sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/amdgpu/$ROCM_APT_VER/ubuntu jammy main > /etc/apt/sources.list.d/amdgpu.list' &&\
    sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/$ROCM_APT_VER jammy main > /etc/apt/sources.list.d/rocm.list'

RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu jammy main universe | tee -a /etc/apt/sources.list" && \
    amdgpu-install -y --usecase=rocm --no-dkms

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
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf amdgpu-install* && \
# Remove unnecessary rocm components that take a lot of space
    apt-get remove -y composablekernel-dev miopen-hip rocfft rocsparse

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Add requirements files
ADD rbuild.ini /rbuild.ini
ADD requirements.txt /requirements.txt
ADD dev-requirements.txt /dev-requirements.txt
ADD docs/sphinx/requirements.txt /doc-requirements.txt

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && rm dumb-init_*.deb && \
# Install cget && rbuild
    pip3 install https://github.com/pfultz2/cget/archive/a426e4e5147d87ea421a3101e6a3beca541c8df8.tar.gz && \
    pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/6d78a0553babdaea8d2da5de15cbda7e869594b8.tar.gz && \
# Add symlink to /opt/rocm
    [ -d /opt/rocm ] || ln -sd $(realpath /opt/rocm-*) /opt/rocm && \
# clear ccachc temp
    rm -rf /tmp/ccache* && mkdir /tmp/ccache && \
# Install selected Ccache commit
    wget -O /tmp/ccache.tar.gz https://github.com/ccache/ccache/archive/${CCACHE_COMMIT}.tar.gz && \
    tar zxvf /tmp/ccache.tar.gz -C /tmp/ && mkdir /tmp/ccache-${CCACHE_COMMIT}/build && \
    cd /tmp/ccache-${CCACHE_COMMIT}/build && \
    cmake -DZSTD_FROM_INTERNET=ON -DHIREDIS_FROM_INTERNET=ON .. && make -j install && rm -rf /tmp/* && \
    ccache -s && \
    cd / && \
# Install doc requirements
    pip3 install -r /doc-requirements.txt && \
# Composable Kernel requires this version cmake
    pip3 install --upgrade cmake==3.27.5 && \
# groupadd render && video
    groupadd -f render && \
    groupadd -f video && \
    usermod -a -G render,video root

# Make sure /opt/rcom is in the paths
ENV PATH="/opt/rocm:${PATH}"

RUN echo Building for GPU Archs: ${GPU_ARCHS} && \
    CK_COMMIT=$(grep 'ROCm/composable_kernel' requirements.txt | sed -n 's/.*@\([a-zA-Z0-9]*\).*/\1/p') && \
    wget -O ck.tar.gz https://www.github.com/ROCm/composable_kernel/archive/${CK_COMMIT}.tar.gz && \
    tar zxvf ck.tar.gz &&\
    cd composable_kernel-${CK_COMMIT} && \
    mkdir build && cd build && \
    num_threads=$(nproc) && \
    if [ "$num_threads" -lt 32 ]; then \
        num_threads=$(( num_threads / 2 )); \
    else \
        num_threads=32; \
    fi && \
    echo Building CK with ${num_threads} threads && \
    CXX=/opt/rocm/bin/amdclang++ cmake \
    -D CMAKE_PREFIX_PATH=/opt/rocm \
    -D CMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D GPU_ARCHS="${GPU_ARCHS}" \
    -D CMAKE_CXX_FLAGS=" -O3 " .. && \
    make -j ${num_threads} install

# Composable Kernel installed separated from rbuild to take in values from GPU_ARCHS 
RUN sed -i '/composable_kernel/d' /requirements.txt

# rbuild is used to trigger build of requirements.txt, dev-requirements.txt
RUN if [ "$USE_FIN" = "ON" ]; then \
        rbuild prepare -s fin -d $PREFIX -DGPU_ARCHS="${GPU_ARCHS}" -DCMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}"; \
    else \
        rbuild prepare -s develop -d $PREFIX -DGPU_ARCHS="${GPU_ARCHS}" -DCMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}"; \
    fi && \
    ccache -s 

#install miopen for perf test builds, remove it if not needed.
# todo: split this out from this dockerfile and move it elsewhere.
ADD . / miopen/
RUN set -e; \
    if [ "$INSTALL_MIOPEN" = "ON" ]; then \
        cd miopen; \
        mkdir build; \
        rm -f src/kernels/*.ufdb.txt; \
        rm -f src/kernels/miopen*.udb; \
        cd build ; \
        CXX=/opt/rocm/llvm/bin/clang++ CXXFLAGS='-Werror'  cmake -DMIOPEN_TEST_FLAGS=' --disable-verification-cache ' -DCMAKE_BUILD_TYPE=release -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm ..; \
        LLVM_PATH=/opt/rocm/llvm CTEST_PARALLEL_LEVEL=4  dumb-init make -j $(nproc) install; \
    else \
        rm -rf miopen; \
    fi

# Utilize multi-stage build in order to squash the container.
FROM ubuntu:22.04
COPY --from=miopen / /

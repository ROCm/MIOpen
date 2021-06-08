def rocmnode(name) {
    return 'rocmtest && miopen && ' + name
}

def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME"
        lsb_release -sd
        uname -r
        cat /sys/module/amdgpu/version
        ls /opt/ -la
    """
}

def cmake_build(compiler, flags, env4make, extradebugflags, prefixpath){
    def workspace_dir = pwd()
    def vcache = "/var/jenkins/.cache/miopen/vcache"
    def archive = (flags == '-DCMAKE_BUILD_TYPE=release')
    def config_targets = "check doc MIOpenDriver"
    def test_flags = "--disable-verification-cache"
    def debug_flags = "-g ${extradebugflags} -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined"
    def compilerpath = ""
    def configargs = ""

    compilerpath = compiler;
    if (prefixpath != "/usr/local") {
        configargs = "-DCMAKE_PREFIX_PATH=${prefixpath}"
    }

    if(!flags.contains('-DBUILD_DEV=On'))
    {
    	config_targets = 'install ' + config_targets
    	flags = '-DCMAKE_INSTALL_PREFIX=../install ' + flags
    }

    if (archive == true) {
        config_targets = "package"
    }
    def cmd = """
        echo \$HSA_ENABLE_SDMA
        ulimit -c unlimited
        cd build
        CXX=${compilerpath} CXXFLAGS='-Werror' cmake ${configargs} -DMIOPEN_TEST_FLAGS='${test_flags}' -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' ${flags} ..
        MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS=1 CTEST_PARALLEL_LEVEL=4 MIOPEN_VERIFY_CACHE_PATH=${vcache} MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 ${env4make} dumb-init make -j\$(nproc) ${config_targets}
    """
    echo cmd
    sh cmd
    // Only archive from master or develop
    if (archive == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}

def buildhipclangjob(Map conf){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        env.CODECOV_TOKEN="aec031be-7673-43b5-9840-d8fb71a2354e"
        checkout scm
        def compiler = conf.get("compiler", "/opt/rocm/llvm/bin/clang++")
        def prefixpath = conf.get("prefixpath", "/usr/local")
        def flags = conf.get("flags", "")
        def env4make = conf.get("env4make", "")
        def image = "miopen"
        def cmd = conf.get("cmd", "")
        def gpu_arch = conf.get("gpu_arch", "gfx900;gfx906")
        def target_id = conf.get("target_id", "OFF")
        def codecov = conf.get("codecov", false)
        def miotensile_version = conf.get("miotensile_version", "default")
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg MIOTENSILE_VER='${miotensile_version}' --build-arg USE_TARGETID='${target_id}' "
        def extradebugflags = ""
        def variant = env.STAGE_NAME
        if (codecov) {
            extradebugflags = "-fprofile-arcs -ftest-coverage"
        }
        def retimage
        gitStatusWrapper(credentialsId: '7126e5fe-eb51-4576-b52b-9aaf1de8f0fd', gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'MIOpen') {
            try {
                retimage = docker.build("${image}", dockerArgs + '.')
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES')
                    {
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }
            catch(Exception ex) {
                retimage = docker.build("${image}", dockerArgs + "--no-cache .")
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES')
                    {
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                    }
                }
            }

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 5, unit: 'HOURS')
                {
                    sh '''
                        rm -rf build
                        mkdir build
                        rm -rf install
                        mkdir install
                        rm -f src/kernels/*.ufdb.txt
                        rm -f src/kernels/miopen*.udb
                    '''
                    if(cmd == ""){
                        cmake_build(compiler, flags, env4make, extradebugflags, prefixpath)
                    }else{
                        sh cmd
                    }
                    if (codecov) {
                        sh '''
                            cd build
                            lcov --directory . --capture --output-file $(pwd)/coverage.info
                            lcov --remove $(pwd)/coverage.info '/usr/*' --output-file $(pwd)/coverage.info
                            lcov --list $(pwd)/coverage.info
                            curl -s https://codecov.io/bash | bash
                            echo "Uploaded"
                        '''
                    }
                }
            }
        }
        return retimage
}

def reboot(){
    build job: 'reboot-slaves', propagate: false , parameters: [string(name: 'server', value: "${env.NODE_NAME}"),]
}

def runDockerJob(Map conf){
    try{
        def build_config = conf
        buildhipclangjob(build_config)
    }
    catch(e){
        echo "throwing error exception for the stage"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        reboot()
    }
}

/// Stage name format:
/// [DataType] Backend[/Compiler] BuildType [TestSet] [Target]
///
/// The only mandatory elements are Backend and BuildType; others are optional.
///
/// DataType := { Fp16 | Bf16 | Int8 | Fp32 }
/// Backend := { Hip | OpenCL | HipNoGPU}
/// Compiler := { Clang* | GCC* }
///   * "Clang" is the default for the Hip backend, and implies hip-clang compiler.
///     For the OpenCL backend, "Clang" implies the system x86 compiler.
///   * "GCC" is the default for OpenCL backend.
///   * The default compiler is usually not specified.
/// BuildType := { Release* | Debug | Install } [ BuildTypeModifier ]
///   * BuildTypeModifier := { COMGR | Embedded | Static | Normal-Find | Fast-Find
///                            MLIR | Tensile | Tensile-Latest | Package | ... }
/// TestSet := { All | Smoke* } [ Codecov ]
///   * "All" corresponds to "cmake -DMIOPEN_TEST_ALL=On".
///   * "Smoke" (-DMIOPEN_TEST_ALL=Off) is the default and usually not specified.
///   * "Codecov" is optional code coverage analysis.
/// Target := { gfx908 | Vega20 | Vega10 | Vega* }
///   * "Vega" (gfx906 or gfx900) is the default and usually not specified.

pipeline {
    agent none
    options {
        parallelsAlwaysFailFast()
    }
    parameters {
        booleanParam(
            name: "STATIC_CHECKS",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "SMOKE_TESTS",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "SMOKE_MLIR",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "SMOKE_MIOPENTENSILE_LATEST",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "FULL_TESTS",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "MIOPENTENSILE",
            defaultValue: false,
            description: "")
        booleanParam(
            name: "MIOPENTENSILE_LATEST",
            defaultValue: false,
            description: "")
        booleanParam(
            name: "PACKAGES",
            defaultValue: true,
            description: "")
    }
    stages{
        stage("Static checks"){
            when { expression { params.STATIC_CHECKS } }
            parallel{
                stage('Hip Tidy') {
                    agent{  label rocmnode("nogpu") }
                    environment{
                        cmd = "cd build; CXX='/opt/rocm/llvm/bin/clang++' cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On ..; make -j\$(nproc) -k analyze;"
                    }
                    steps{
                        script{
                            try{
                                buildhipclangjob(compiler: 'clang++', flags: '-DCMAKE_BUILD_TYPE=release', cmd: cmd)
                            }
                            catch(e){
                                echo "throwing error exception for the stage"
                                echo 'Exception occurred: ' + e.toString()
                                throw e
                            }
                        }
                    }
                }
                stage('OpenCL Tidy') {
                    agent{  label rocmnode("nogpu") }
                    environment{
                        cmd = "cd build; cmake -DMIOPEN_BACKEND=OpenCL -DBUILD_DEV=On ..; make -j\$(nproc) -k analyze;"
                    }
                    steps{
                        script{
                            try{
                                buildhipclangjob(compiler: 'g++', flags: '-DCMAKE_BUILD_TYPE=release', cmd: cmd)
                            }
                            catch(e){
                                echo "throwing error exception for the stage"
                                echo 'Exception occurred: ' + e.toString()
                                throw e
                            }
                        }
                    }
                }
                stage('Clang Format') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        cmd = "find . -iname \'*.h\' \
                                -o -iname \'*.hpp\' \
                                -o -iname \'*.cpp\' \
                                -o -iname \'*.h.in\' \
                                -o -iname \'*.hpp.in\' \
                                -o -iname \'*.cpp.in\' \
                                -o -iname \'*.cl\' \
                                | grep -v 'build/' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-3.8 -style=file {} | diff - {}\'"
                    }
                    steps{
                        script{
                            try{
                                buildhipclangjob(compiler: 'clang++', flags: '-DCMAKE_BUILD_TYPE=release', cmd: cmd)
                            }
                            catch(e){
                                echo "throwing error exception for the stage"
                                echo 'Exception occurred: ' + e.toString()
                                throw e
                            }
                        }
                    }
                }
            }
        }
        stage("Smoke Fp32"){
            when { expression { params.SMOKE_TESTS } }
            parallel{
               stage('Fp32 OpenCL Debug') {
                    agent{ label rocmnode("vega") }
                    steps{
                        script{
                            runDockerJob(compiler: 'g++', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
                        }
                    }
                }
                stage('Fp32 OpenCL') {
                    agent{ label rocmnode("vega") }
                    steps{
                        script{
                            runDockerJob(compiler: 'g++', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
                        }
                    }
                }
                stage('Fp32 Hip') {
                    agent{ label rocmnode("vega") }
                    steps{
                        script{
                            runDockerJob(flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
                        }
                    }
                }
                stage('Fp32 Hip /opt/rocm') {
                    agent{ label rocmnode("vega") }
                    steps{
                        script{
                            runDockerJob(flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', prefixpath: '/opt/rocm')
                        }
                    }
                }
                stage('Fp32 Hip Debug') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd)
                        }
                    }
                }
                stage('Fp32 Hip Debug gfx908 /opt/rocm') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        script{
                            runDockerJob(flags: '-DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', prefixpath: '/opt/rocm', gpu_arch: "gfx908")
                        }
                    }
                }
            }
        }
        stage("Smoke Aux 1"){
            when { expression { params.SMOKE_TESTS } }
            parallel{
                stage('Fp32 HipNoGPU Debug') {
                    agent{  label rocmnode("nogpu") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_BACKEND=HIPNOGPU -DMIOPEN_INSTALL_CXX_HEADERS=On ..
                            make -j\$(nproc)
                        """
                    }
                    steps{
                        buildhipclangjob(env4make: "MIOPEN_LOG_LEVEL=5 MIOPEN_COMPILE_PARALLEL_LEVEL=1", cmd: cmd)
                    }
                }
                stage('Fp32 Hip Debug COMGR') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_USE_COMGR=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            CTEST_PARALLEL_LEVEL=2 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd)
                        }
                    }
                }
                stage('Fp32 Hip Debug Embedded Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_EMBED_DB="gfx906_60;gfx906_64" -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd)
                        }
                    }
                }
                stage('Fp32 Hip Static') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DBUILD_SHARED_LIBS=Off -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd)
                        }
                    }
                }
                stage('Fp32 Hip Normal-Find') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release ..
                            make -j test_conv2d
                            MIOPEN_FIND_MODE=normal CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --disable-verification-cache
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd)
                        }
                    }
                }
                stage('Fp32 Hip Fast-Find') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release ..
                            make -j test_conv2d
                            MIOPEN_FIND_MODE=fast CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --disable-verification-cache
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd)
                        }
                    }
                }
            }
        }
        stage("Smoke MLIR"){
            when { expression { params.SMOKE_MLIR } }
            parallel{
                stage('Fp32 Hip MLIR') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_USE_MLIR=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx900;gfx906")
                        }
                    }
                }
                stage('Fp16 Hip MLIR') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_HALF=On -DMIOPEN_USE_MLIR=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx900;gfx906")
                        }
                    }
                }
                stage('Fp32 Hip MLIR Xdlops') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_USE_MLIR=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908")
                        }
                    }
                }
                stage('Fp16 Hip MLIR Xdlops') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake  -DMIOPEN_TEST_HALF=On -DMIOPEN_USE_MLIR=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908")
                        }
                    }
                }
            }
        }
        stage("Smoke Fp16/Bf16/Int8"){
            when { expression { params.SMOKE_TESTS } }
            parallel{
                stage('Fp16 Hip Vega20 /opt/rocm') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        script{
                            runDockerJob(flags: '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', prefixpath: '/opt/rocm')
                        }
                    }
                }
                stage('Fp16 OpenCL Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        script{
                            runDockerJob(compiler: 'g++', flags: '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
                        }
                    }
                }
                stage('Int8 OpenCL Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        script{
                            runDockerJob(compiler: 'g++', flags: '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
                        }
                    }
                }
                stage('Bf16 Hip Vega20 /opt/rocm') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        script{
                            runDockerJob(flags: '-DMIOPEN_TEST_BFLOAT16=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', prefixpath: '/opt/rocm')
                        }
                    }
                }
                stage('Bf16 Hip Debug gfx908 /opt/rocm') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        script{
                            runDockerJob(flags: '-DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', prefixpath: '/opt/rocm', gpu_arch: "gfx908")
                        }
                    }
                }
                stage('Fp16 Hip Debug gfx908 /opt/rocm') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        script{
                            runDockerJob(flags: '-DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', prefixpath: '/opt/rocm', gpu_arch: "gfx908")
                        }
                    }
                }
            }
        }
        stage("Smoke MIOpenTensile Latest"){
            when { expression { params.SMOKE_MIOPENTENSILE_LATEST } }
            parallel{
                stage('Fp16 Hip Tensile-Latest Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_HALF=On -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx906:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
                stage('Int8 Hip Tensile-Latest Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx906:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
                stage('Fp32 Hip Tensile-Latest gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
                stage('Bf16 Hip Tensile-Latest gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
            }
        }
        stage("Full tests I"){
            when { expression { params.FULL_TESTS } }
            parallel{
                stage('Fp32 OpenCL Debug + Codecov') {
                    agent{ label rocmnode("vega") }
                    steps{
                        script{
                            runDockerJob(compiler: 'g++', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', codecov: true)
                        }
                    }
                }
                stage('Int8 Hip All Vega20 /opt/rocm') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        script{
                            runDockerJob(flags: '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', prefixpath: '/opt/rocm')
                        }
                    }
                }
                stage('Bf16 Hip Install All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) install check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908")
                        }
                    }
                }
            }
        }
        stage("Full tests II"){
            when { expression { params.FULL_TESTS } }
            parallel{
                stage('Fp32 OpenCL Install All') {
                    agent{ label rocmnode("vega") }
                    steps{
                        script{
                            runDockerJob(compiler: 'g++', flags: '-DBUILD_DEV=Off -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
                        }
                    }
                }
                stage('Fp32 Hip All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908")
                        }
                    }
                }
                stage('Fp16 Hip Install All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_HALF=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) install check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908")
                        }
                    }
                }
            }
        }
        stage("Full tests III"){
            when { expression { params.FULL_TESTS } }
            parallel{
                stage('Fp16 Hip Install All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_HALF=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) install check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd)
                        }
                    }
                }
                stage('Fp32 Hip All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd)
                        }
                    }
                }
            }
        }
        stage("MIOpenTensile"){
            when { expression { params.MIOPENTENSILE } }
            parallel{
                stage('Fp32 Hip Tensile All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx906:xnack-", miotensile_version: "default", target_id: "ON")
                        }
                    }
                }
                stage('Fp16 Hip Tensile All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_HALF=On -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx906:xnack-", miotensile_version: "default", target_id: "ON")
                        }
                    }
                }
                stage('Bf16 Hip Tensile All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx906:xnack-", miotensile_version: "default", target_id: "ON")
                        }
                    }
                }
                stage('Int8 Hip Tensile All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_INT8=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx906:xnack-", miotensile_version: "default", target_id: "ON")
                        }
                    }
                }
                stage('Fp32 Hip Tensile All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908:xnack-", miotensile_version: "default", target_id: "ON")
                        }
                    }
                }
                stage('Fp16 Hip Tensile All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_HALF=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908:xnack-", miotensile_version: "default", target_id: "ON")
                        }
                    }
                }
                stage('Bf16 Hip Tensile All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908:xnack-", miotensile_version: "default", target_id: "ON")
                        }
                    }
                }
                stage('Int8 Hip Tensile All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_INT8=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908:xnack-", miotensile_version: "default", target_id: "ON")
                        }
                    }
                }
            }
        }
        stage("MIOpenTensile Latest"){
            when { expression { params.MIOPENTENSILE_LATEST } }
            parallel{
                stage('Fp32 Hip Tensile-Latest All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx906:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
                stage('Fp16 Hip Tensile-Latest All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_HALF=On -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx906:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
                stage('Bf16 Hip Tensile-Latest All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx906:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
                stage('Int8 Hip Tensile-Latest All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_INT8=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx906:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
                stage('Fp32 Hip Tensile-Latest All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
                stage('Fp16 Hip Tensile-Latest All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_HALF=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
                stage('Bf16 Hip Tensile-Latest All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
                stage('Int8 Hip Tensile-Latest All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_INT8=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        script{
                            runDockerJob(cmd: cmd, gpu_arch: "gfx908:xnack-", miotensile_version: "latest", target_id: "ON")
                        }
                    }
                }
            }
        }
        stage("Packages"){
            when { expression { params.PACKAGES } }
            parallel {
                stage('OpenCL Package') {
                    agent{ label rocmnode("nogpu") }
                    steps{
                        script{
                            runDockerJob(compiler: 'g++', flags: '-DCMAKE_BUILD_TYPE=release', gpu_arch: "gfx900;gfx906;gfx908")
                        }
                    }
                }
                stage("HIP Package /opt/rocm"){
                    agent{ label rocmnode("nogpu") }
                    steps{
                        script{
                            runDockerJob(flags: '-DCMAKE_BUILD_TYPE=release', prefixpath: '/opt/rocm', gpu_arch: "gfx900;gfx906;gfx908")
                        }
                    }
                }
            }
        }
    }
}

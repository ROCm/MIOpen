
def rocmnode(name) {
    def node_name = 'rocmtest'
    if(name == 'fiji') {
        node_name = 'rocmtest && fiji';
    } else if(name == 'vega') {
        node_name = 'rocmtest && vega';
    } else if(name == 'vega10') {
        node_name = 'rocmtest && vega10';
    } else if(name == 'vega20') {
        node_name = 'rocmtest && vega20';
    } else if(name == 'gfx908') {
        node_name = 'gfx908';
    } else {
        node_name = name
    }
    return node_name
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
    if (prefixpath == "/usr/local")
        compilerpath = compiler;
    else
    {
        compilerpath = prefixpath + "/bin/" + compiler
        configargs = "-DCMAKE_PREFIX_PATH=${prefixpath}"
    }

    if (archive == true) {
        config_targets = "package"
    }
    def cmd = """
        echo \$HSA_ENABLE_SDMA
        ulimit -c unlimited
        rm -rf build
        mkdir build
        cd build
        CXX=${compilerpath} CXXFLAGS='-Werror' cmake ${configargs} -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS='${test_flags}' -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' ${flags} ..
        MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS=1 CTEST_PARALLEL_LEVEL=4 MIOPEN_VERIFY_CACHE_PATH=${vcache} MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 ${env4make} dumb-init make -j\$(nproc) ${config_targets}
    """
    echo cmd
    sh cmd
    // Only archive from master or develop
    if (archive == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}

def buildJob(Map conf, compiler){

        env.HSA_ENABLE_SDMA=0
        env.CODECOV_TOKEN="aec031be-7673-43b5-9840-d8fb71a2354e"
        checkout scm
        def prefixpath = conf.get("prefixpath", "/usr/local")
        def flags = conf.get("flags", "")
        def env4make = conf.get("env4make", "")
        def image = conf.get("image", "miopen")
        def cmd = conf.get("cmd", "")
        def gpu_arch = conf.get("gpu_arch", "all")
        def codecov = conf.get("codecov", false)
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' "
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
                        sh 'PATH="/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                    }
                }
            } catch(Exception ex) {
                retimage = docker.build("${image}", dockerArgs + "--no-cache .")
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES')
                    {
                        sh 'PATH="/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                    }
                }
            }

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 5, unit: 'HOURS')
                {
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

def buildHipClangJob(compiler, flags, env4make, image, prefixpath="/opt/rocm", cmd = "", gpu_arch="all", miot_ver="default"){

        env.HSA_ENABLE_SDMA=0
        checkout scm
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg MIOTENSILE_VER='${miot_ver}' -f hip-clang.docker "
        def variant = env.STAGE_NAME
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
            } catch(Exception ex) {
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
                    if(cmd == ""){
                        cmake_build(compiler, flags, env4make, prefixpath)
                    }else{
                        sh cmd
                    }
                }
            }
        }
        return retimage
}

/// Stage name format:
/// [DataType] Backend[/Compiler] BuildType [TestSet] [Target]
///
/// The only mandatory elements are Backend and BuildType; others are optional.
///
/// DataType := { Half | BF16 | Int8 | FP32* }
///   * "FP32" is the default and usually not specified.
/// Backend := { Hip | OpenCL }
/// Compiler := { Clang* | hcc | GCC* }
///   * "Clang" is the default for the Hip backend, and implies hip-clang compiler.
///     For the OpenCL backend, "Clang" implies the system x86 compiler.
///   * "GCC" is the default for OpenCL backend.
///   * The default compiler is usually not specified.
/// BuildType := { Release | Debug [ BuildTypeModifier ] }
///   * BuildTypeModifier := { COMGR | Embedded | Static | Normal-Find | Fast-Find
///                                  | Tensile | Tensile-Latest | Package | ... }
/// TestSet := { All | Subset | Smoke* }
///   * "All" corresponds to "cmake -DMIOPEN_TEST_ALL=On".
///   * "Subset" corresponds to Target- or BuildTypeModifier-specific subsetting of
///     the "All" testset, e.g. -DMIOPEN_TEST_GFX908=On or -DMIOPEN_TEST_MIOTENSILE=On.
///   * "Smoke" (-DMIOPEN_TEST_ALL=Off) is the default and usually not specified.
/// Target := { gfx908 | Vega20 | Vega10 | Fiji | Vega* }
///   * "Vega" (gfx906 or gfx900) is the default and usually not specified.

pipeline {
    agent none
    options {
        parallelsAlwaysFailFast()
    }
    environment{
        image = "miopen"
    }
    stages{
        // Run all static analysis tests
        stage("Static checks"){
            parallel{
                stage('Clang Tidy') {
                    agent{  label rocmnode("rocmtest") }
                    environment{
                        cmd = "rm -rf build; mkdir build; cd build; CXX='clang++-3.8' cmake -DBUILD_DEV=On ..; make -j\$(nproc) -k analyze;"
                    }
                    steps{
                        buildJob('hcc', flags: '-DCMAKE_BUILD_TYPE=release', cmd: cmd, gpu_arch: "all")
                    }
                }

                stage('Clang Format') {
                    agent{ label rocmnode("rocmtest") }
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
                        buildJob('hcc', flags: '-DCMAKE_BUILD_TYPE=release', cmd: cmd, gpu_arch: "all")
                    }
                }

                stage('Hip Tidy') {
                    agent{ label rocmnode("rocmtest") }
                    environment{
                        cmd = "rm -rf build; mkdir build; cd build; CXX=/usr/local/bin/hcc cmake -DBUILD_DEV=On ..; make -j\$(nproc) -k analyze;"
                    }
                    steps{
                        buildJob('hcc', flags: '-DCMAKE_BUILD_TYPE=release', cmd: cmd, gpu_arch: "all")
                    }
                }
            }
        }

        // Run quick fp32 tests
        stage("Fast full precision"){
            parallel{
               stage('OpenCL/Clang Debug') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('clang++-3.8', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', gpu_arch: "gfx900;gfx906")
                    }
                }

                stage('OpenCL/Clang Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('clang++-3.8', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', gpu_arch: "gfx900;gfx906")
                    }
                }

                stage('OpenCL Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('g++-5', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', gpu_arch: "gfx900;gfx906")
                    }
                }

                stage('OpenCL Debug Fiji') {
                    agent{ label rocmnode("fiji") }
                    steps{
                        buildJob('g++-5', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', gpu_arch: "gfx803")
                    }
                }

                stage('Hip/hcc Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('hcc', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image: image + "rocm", gpu_arch: "gfx900;gfx906")
                    }
                }

                stage('Hip Debug') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "", image+'-hip-clang', "/usr/local", cmd, "gfx900;gfx906")
                    }
                }



                stage('Hip/hcc Debug gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildJob('hcc', flags: '-DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', image: image+"rocm", prefixpath: '/opt/rocm', gpu_arch: "gfx908")
                    }
                }
            }
        }

        // Misc tests
        stage("Aux tests"){
            parallel{
                stage('Hip Debug COMGR') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_USE_COMGR=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=2 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "MIOPEN_LOG_LEVEL=5 MIOPEN_COMPILE_PARALLEL_LEVEL=1",  image+'-hip-clang', "/usr/local", cmd, "gfx900;gfx906")
                    }
                }
                stage('Hip Debug Embedded Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_EMBED_DB="gfx906_60;gfx906_64" -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "MIOPEN_LOG_LEVEL=5 MIOPEN_COMPILE_PARALLEL_LEVEL=1",  image+'-hip-clang', "/usr/local", cmd, "gfx906")
                    }
                }

                stage('Hip Release Static') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DBUILD_SHARED_LIBS=Off -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "",  image+'-hip-clang', "/usr/local", cmd, "gfx900;gfx906")
                    }
                }

                stage('Hip Release Normal-Find') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On ..
                            make -j test_conv2d
                            MIOPEN_FIND_MODE=1 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --disable-verification-cache
                        """
                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "",  image+'-hip-clang', "/usr/local", cmd, "gfx900;gfx906")
                    }
                }

                stage('Hip Release Fast-Find') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On ..
                            make -j test_conv2d
                            MIOPEN_FIND_MODE=2 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --disable-verification-cache
                        """
                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "",  image+'-hip-clang', "/usr/local", cmd, "gfx900;gfx906")
                    }
                }

                stage('Hip/hcc Release on /usr/local') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('hcc', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', gpu_arch: "gfx900;gfx906")
                    }
                }

            }
        }

        // Run fp16, bfp16, and int8 quick tests
        stage("Fast low precision"){
            parallel{
                stage('Half Hip/hcc Release Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', flags: '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image: image+"rocm", prefixpath: '/opt/rocm', gpu_arch: "gfx906")
                    }
                }
                stage('Half OpenCL Release Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('g++-5', flags: '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', gpu_arch: "gfx906")
                    }
                }
                stage('Int8 OpenCL Release Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('g++-5', flags: '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', gpu_arch: "gfx906")
                    }
                }

                stage('BF16 Hip/hcc Release Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', flags: '-DMIOPEN_TEST_BFLOAT16=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image: image+"rocm", prefixpath: '/opt/rocm', gpu_arch: "gfx906")
                    }
                }

                stage('BF16 Hip/hcc Debug gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildJob('hcc', flags: '-DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', image: image+"rocm", prefixpath: '/opt/rocm', gpu_arch: "gfx908")
                    }
                }
                stage('Half Hip/hcc Debug gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildJob('hcc', flags: '-DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', image: image+"rocm", prefixpath: '/opt/rocm', gpu_arch: "gfx908")
                    }
                }
            }
        }

        stage("Full tests I"){
            parallel{

                stage('GCC codecov') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('g++-5', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', codecov: true, gpu_arch: "gfx900;gfx906")
                    }
                }

                stage('Int8 Hip/hcc Release All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', flags: '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', image: image+"rocm", prefixpath: '/opt/rocm', gpu_arch: "gfx906")
                    }
                }

                stage('BF16 Hip Release Subset gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On ..
                            MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "",  image+'-hip-clang', "/usr/local", cmd, "gfx908")
                    }
                }
            }
        }

        stage("Full tests II"){
            parallel{
                stage('OpenCL Release All') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('g++-5', flags: '-DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', gpu_arch: "gfx900;gfx906")
                    }
                }

                stage('Hip Release Subset gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On ..
                            MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "",  image+'-hip-clang', "/usr/local", cmd, "gfx908")
                    }
                }

                stage('Half Hip Release Subset gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_TEST_HALF=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On ..
                            MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "",  image+'-hip-clang', "/usr/local", cmd, "gfx908")
                    }
                }
            }
        }

        stage("Full tests III"){
            parallel{
                stage('Half Hip Release All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_HALF=On -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "", image+'-hip-clang', "/usr/local", cmd, "gfx906")
                    }
                }

                stage('Hip Release All Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "", image+'-hip-clang', "/usr/local", cmd, "gfx906")
                    }
                }
            }
        }

        stage("MIOpenTensile"){
            parallel{
                stage('Hip Release Tensile Subset Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "", image+'-hip-clang', "/usr/local", cmd, "gfx906")
                    }
                }

                stage('Hip Release Tensile Subset gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "", image+'-hip-clang', "/usr/local", cmd, "gfx908")
                    }
                }

                stage('Hip Release Tensile-Latest Subset Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS=--disable-verification-cache ..
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "", image+'-hip-clang', "/usr/local", cmd, "gfx906", "latest")
                    }
                }

                stage('Hip Release Tensile-Latest Subset gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF ..
                            MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """
                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "", image+'-hip-clang', "/usr/local", cmd, "gfx908", "latest")
                    }
                }
            }
        }

       // Run package building
        stage("Packages"){
            parallel {
                stage('OpenCL Release Package') {
                    agent{ label rocmnode("rocmtest") }
                    steps{
                        buildJob('g++-5', flags: '-DCMAKE_BUILD_TYPE=release', gpu_arch: "all")
                    }
                }
                stage("HIP/hcc Release Package"){
                    agent{ label rocmnode("rocmtest") }
                    steps{
                        buildJob('hcc', flags: '-DCMAKE_BUILD_TYPE=release', image: image+"rocm", prefixpath: '/opt/rocm', gpu_arch: "all")
                    }
                }
            }
        }
    }
}


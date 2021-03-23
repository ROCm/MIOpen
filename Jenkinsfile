
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

def cmake_build(Map conf){

    def compiler = conf.get("compiler","/opt/rocm/llvm/bin/clang++")
    def setup_args = "-DMIOPEN_GPU_SYNC=On "
    
    def prefixpath = conf.get("prefixpath","/usr/local")

    if (prefixpath != "/usr/local") {
        setup_args = setup_args + "-DCMAKE_PREFIX_PATH=${prefixpath} "
    }

    def config_targets = conf.get("config_targets","check doc MIOpenDriver")
    def package_build = (conf.get("package_build","") == "true")
    
    if (package_build == true) {
        config_targets = "package"
    }
    
    def build_envs = "CTEST_PARALLEL_LEVEL=4" + conf.get("build_env"," MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS=1")
    
    def test_flags = conf.get("test_flags","")
    if (conf.get("vcache_enable","") == "true"){
        def vcache = conf.get(vcache_path,"/var/jenkins/.cache/miopen/vcache")
        build_envs = "MIOPEN_VERIFY_CACHE_PATH=${vcache} " + build_envs
    } else{
        test_flags = "--disable-verification-cache " + test_flags
    }

    if(test_flags){
        setup_args = ${setup_args} + "-DMIOPEN_TEST_FLAGS='${test_flags}' "
    }
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined" + conf.get("extradebugflags", "")
    setup_args = ${setup_args} + "-DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' " + conf.get("flags","")

    //cmake_env can overwrite default CXX variables. 
    def cmake_envs = "CXX=${compiler} CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")

    def pre_setup_cmd = """
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            rm -rf build
            mkdir build
            cd build
        """
    def setup_cmd = conf.get("setup_cmd", "${cmake_envs} cmake ${setup_args} .. ")
    def build_cmd = conf.get("build_cmd", "${build_envs} dumb-init make -j\$(nproc) ${config_targets}")
    def execute_cmd = conf.get("build_cmd", "")

    def cmd = conf.get("cmd", """
            ${pre_setup_cmd}
            ${setup_cmd}
            ${build_cmd}
            ${execute_cmd}
        """)

    echo cmd
    sh cmd

    // Only archive from master or develop
    if (package_build == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}

def buildHipClangJob(Map conf){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        env.CODECOV_TOKEN="aec031be-7673-43b5-9840-d8fb71a2354e"
        checkout scm

        def image = "miopen"
        def prefixpath = conf.get("prefixpath", "/usr/local")
        def gpu_arch = conf.get("gpu_arch", "gfx900;gfx906")

        def miotensile_version = conf.get("miotensile_version", "default")
        
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg MIOTENSILE_VER='${miotensile_version}' "

        def variant = env.STAGE_NAME

        def codecov = conf.get("codecov", false)
        if (codecov) {
            conf["extradebugflags"] = "-fprofile-arcs -ftest-coverage" + conf.get("extradebugflags", "")
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
                    cmake_build(conf)

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

def buildHipClangJobAndReboot(Map conf){
    try{
        buildHipClangJob(conf)
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
/// DataType := { Half | BF16 | Int8 | FP32* }
///   * "FP32" is the default and usually not specified.
/// Backend := { Hip | OpenCL | HipNoGPU}
/// Compiler := { Clang* | GCC* }
///   * "Clang" is the default for the Hip backend, and implies hip-clang compiler.
///     For the OpenCL backend, "Clang" implies the system x86 compiler.
///   * "GCC" is the default for OpenCL backend.
///   * The default compiler is usually not specified.
/// BuildType := { Release | Debug [ BuildTypeModifier ] }
///   * BuildTypeModifier := { COMGR | Embedded | Static | Normal-Find | Fast-Find
///                                  | Tensile | Tensile-Latest | Package | ... }
/// TestSet := { All | Subset | Smoke* } [ Codecov ]
///   * "All" corresponds to "cmake -DMIOPEN_TEST_ALL=On".
///   * "Subset" corresponds to Target- or BuildTypeModifier-specific subsetting of
///     the "All" testset, e.g. -DMIOPEN_TEST_GFX908=On or -DMIOPEN_TEST_MIOTENSILE=On.
///   * "Smoke" (-DMIOPEN_TEST_ALL=Off) is the default and usually not specified.
///   * "Codecov" is optional code coverage analysis.
/// Target := { gfx908 | Vega20 | Vega10 | Vega* }
///   * "Vega" (gfx906 or gfx900) is the default and usually not specified.


pipeline {
    agent none
    options {
        parallelsAlwaysFailFast()
    }

    stages{
        // Run all static analysis tests

        stage("Static checks"){
            parallel{
                stage('Clang Hip Tidy') {
                    agent{  label rocmnode("nogpu") }
                    environment{
                        cmd = "rm -rf build; mkdir build; cd build; CXX='/opt/rocm/llvm/bin/clang++' cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On ..; make -j\$(nproc) -k analyze;"
                    }
                    steps{
                        script{
                            try{
                                buildHipClangJob(cmd: cmd)
                            }
                            catch(e){
                                echo "throwing error exception for the stage"
                                echo 'Exception occurred: ' + e.toString()
                                throw e
                            }
                        }
                    }
                }
                stage('Clang OpenCL Tidy') {
                    agent{  label rocmnode("nogpu") }
                    environment{
                        cmd = "rm -rf build; mkdir build; cd build; CXX='clang++-3.8' cmake -DMIOPEN_BACKEND=OpenCL -DBUILD_DEV=On ..; make -j\$(nproc) -k analyze;"
                    }
                    steps{
                        script{
                            try{
                                buildHipClangJob(cmd: cmd)
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
                                buildHipClangJob(cmd: cmd)
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

        // Run quick fp32 tests
        stage("Fast full precision"){
            parallel{
               stage('OpenCL Debug') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
                    }
                }

                stage('OpenCL Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
                    }
                }

                stage('Hip conv2d debug quick') {
                     agent{ label rocmnode("vega") }
                     environment{
                        setup_cmd = "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .."
                        build_cmd = "make -j\$(nproc) test_conv2d"
                        execute_cmd = "bin/test_conv2d --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot( prefixpath: "/usr/local", setup_cmd: setup_cmd, build_cmd: build_cmd, execute_cmd: execute_cmd)
                    }
                }

                stage('Hip release') {
                    agent{ label rocmnode("vega") }
                    environment{
                        setup_cmd = "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS='--disable-verification-cache'  .."
                        build_cmd = "CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: setup_cmd, build_cmd: build_cmd)
                    }
                }

                stage('Hip Debug') {
                    agent{ label rocmnode("vega") }
                    environment{
                        setup_cmd = "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .."
                        build_cmd = "CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: setup_cmd, build_cmd: build_cmd)
                    }
                }

                stage('Hip Debug gfx908 /opt/rocm') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        setup_cmd = "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS='--disable-verification-cache ' .. "
                        build_cmd = "CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                        gpu_arch = "gfx908"
                        prefixpath = "/opt/rocm"
                    }
                    steps{
                        buildHipClangJobAndReboot(prefixpath: prefixpath, gpu_arch: gpu_arch, setup_cmd: setup_cmd, build_cmd: build_cmd)
                        //flags: '-DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', image: image+'rocm', prefixpath: '/opt/rocm', gpu_arch: "gfx908")
                    }
                }
            }
        }

        // Misc tests
        stage("Aux tests"){
            parallel{
                stage('HipNoGPU Debug') {
                    agent{  label rocmnode("nogpu") }
                    environment{
                        setup_cmd = "CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_BACKEND=HIPNOGPU -DMIOPEN_INSTALL_CXX_HEADERS=On .."
                        build_cmd = "make -j\$(nproc)"
                    }
                    steps{
                        buildHipClangJob( setup_cmd: setup_cmd, build_cmd: build_cmd)
                    }
                }

                stage('Hip Debug COMGR') {
                    agent{ label rocmnode("vega") }
                    environment{
                        setup_cmd = "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_USE_COMGR=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' .."
                        build_cmd = "CTEST_PARALLEL_LEVEL=2 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd)
                    }
                }

                stage('Hip Debug Embedded Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        setup_cmd = "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_EMBED_DB='gfx906_60;gfx906_64' -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' .."
                        build_cmd = "MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd)
                    }
                }

                stage('Hip Release Static') {
                    agent{ label rocmnode("vega") }
                    environment{
                        setup_cmd = "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DBUILD_EMBED_BUILD=On  -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .. "
                        build_cmd = "CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd)
                    }
                }

                stage('Hip Release Normal-Find') {
                    agent{ label rocmnode("vega") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On .. "
                        build_cmd =   "make -j test_conv2d"
                        execute_cmd = "MIOPEN_FIND_MODE=1 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, execute_cmd: execute_cmd)
                    }
                }

                stage('Hip Release Fast-Find') {
                    agent{ label rocmnode("vega") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On .. "
                        build_cmd =   "make -j test_conv2d"
                        execute_cmd = "MIOPEN_FIND_MODE=2 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --disable-verification-cache"
                    }
                    steps{
                          buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, execute_cmd: execute_cmd)
                    }
                }

                stage('Hip Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot( flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
                    }
                }

            }
        }

        // Run fp16, bfp16, and int8 quick tests
        stage("Fast low precision"){
            parallel{
                stage('Half Hip Release Vega20 /opt/rocm') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        flags = '-DMIOPEN_BACKEND=HIP -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 -DMIOPEN_TEST_HALF=On -DMIOPEN_USE_COMGR=Off -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release'
                    }
                    steps{
                        buildHipClangJobAndReboot( flags: flags, prefixpath: '/opt/rocm')
                    }
                }

                stage('Half gfx908 Release') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DAMDGPU_TARGETS=gfx908 -DGPU_TARGETS=gfx908 -DMIOPEN_TEST_HALF=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_USE_COMGR=Off -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=Release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS='--disable-verification-cache ' .. "
                        build_cmd =   "CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, gpu_arch: "gfx908")
                    }
                }

                stage('BF16 Hip Debug gfx908 /opt/rocm') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        flags = '-DMIOPEN_BACKEND=HIP -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 -DMIOPEN_TEST_HALF=On -DMIOPEN_USE_COMGR=Off -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release'
                    }
                    steps{
                        buildHipClangJobAndReboot( flags: flags, prefixpath: '/opt/rocm', gpu_arch: "gfx908")
                    }
                }
            }
        }

        stage("Long Tests I"){
            parallel{
                stage('Int8 conv2d Release Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DMIOPEN_TEST_INT8=On -DMIOPEN_USE_COMGR=Off -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=Release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_LIMIT=2 -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .. "
                        build_cmd =   "make -j test_conv2d"
                        execute_cmd = "MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --limit 3 --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, execute_cmd: execute_cmd)
                    }
                }

                stage('GCC Release conv2d') {
                    agent{ label rocmnode("vega") }
                    environment{
                        setup_cmd =   "CXX=/usr/bin/g++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DMIOPEN_BACKEND=OpenCL -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .. "
                        build_cmd =   "make -j test_conv2d "
                        execute_cmd = "CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --limit 3 --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, execute_cmd: execute_cmd, gpu_arch: "gfx908")
                    }
                }

                stage('GCC OpenCL All + codecov') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', flags: '-DBUILD_DEV=On -DMIOPEN_BACKEND=OpenCL -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', codecov: true)
                    }
                }
            }
        }

        stage("Long Tests II"){
            parallel{
                stage('FP32 gfx908 Hip Release All subset') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_LIMIT=2 -DMIOPEN_GPU_SYNC=On -DMIOPEN_USE_COMGR=Off -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .. "
                        build_cmd =   "CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, gpu_arch: "gfx908")
                    }
                }

                stage('Half Hip Clang Release All') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_HALF=On -DMIOPEN_GPU_SYNC=On -DMIOPEN_USE_COMGR=Off -DMIOPEN_TEST_LIMIT=2 -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .. "
                        build_cmd =   "CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, gpu_arch: "gfx908")
                    }
                }
            }
        }

        stage("Long Tests III"){
            parallel{
                stage('Hip Clang conv3d') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=Off -DMIOPEN_USE_COMGR=Off -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_LIMIT=2 .. "
                        build_cmd =   "make -j test_conv3d"
                        execute_cmd = "bin/test_conv3d --all --limit=2 --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: setup_cmd, build_cmd: build_cmd, execute_cmd: execute_cmd )
                    }
                }

                stage('Hip Clang conv2d') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DMIOPEN_SKIP_ALL_BUT_CONV2D=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_USE_COMGR=Off -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_LIMIT=2 -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .. "
                        build_cmd =   "make -j\$(nproc) check "
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd)
                    }
                }

                stage('Half gfx908 Hip Release All Subset') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_TEST_GFX908=On -DMIOPEN_USE_COMGR=Off -DMIOPEN_TEST_HALF=On -DMIOPEN_TEST_LIMIT=2 -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .. "
                        build_cmd =   "CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, gpu_arch: "gfx908")
                    }
                }
            }
        }


        stage("MIOpenTensile"){
            parallel{
                stage('Hip Release Tensile Subset Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .."
                        build_cmd =   "MIOPEN_DEBUG_HIP_KERNELS=0 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                            buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd)
                    }
                }

                stage('Hip Release Tensile Subset gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On  -DMIOPEN_TEST_LIMIT=2 -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' .."
                        build_cmd =   "MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                            buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, gpu_arch: "gfx908")
                    }
                }

                stage('Hip Release Tensile-Latest Subset Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_ALL=On  -DMIOPEN_TEST_LIMIT=2 -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS='--disable-verification-cache' .."
                        build_cmd =   "MIOPEN_DEBUG_HIP_KERNELS=0 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, miotensile_version: "latest")
                    }
                }

                stage('Hip Release Tensile-Latest Subset gfx908') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        setup_cmd =   "CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_GFX908=On -DMIOPEN_TEST_ALL=On  -DMIOPEN_TEST_LIMIT=2 -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF -DMIOPEN_TEST_FLAGS='--verbose --disable-verification-cache' .."
                        build_cmd =   "MIOPEN_DEBUG_HIP_KERNELS=0 MIOPEN_LOG_LEVEL=5 CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( setup_cmd: setup_cmd, build_cmd: build_cmd, gpu_arch: "gfx908", miotensile_version: "latest")
                    }
                }
            }
        }

       // Run package building
        stage("Packages"){
            parallel {
                stage('OpenCL Release Package') {
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', package_build: "true", flags: '-DCMAKE_BUILD_TYPE=release', gpu_arch: "gfx900;gfx906;gfx908")
                    }
                }
                stage("HIP Release Package /opt/rocm"){
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildHipClangJobAndReboot( package_build:"true", flags: '-DCMAKE_BUILD_TYPE=release', prefixpath: '/opt/rocm', gpu_arch: "gfx900;gfx906;gfx908")
                    }
                }
            }
        }
    }
}


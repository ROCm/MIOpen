

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



def cmake_build(compiler, flags, env4make, prefixpath){
    def workspace_dir = pwd()
    def vcache = "/var/jenkins/.cache/miopen/vcache"
    def archive = (flags == '-DCMAKE_BUILD_TYPE=release')
    def config_targets = "check doc MIOpenDriver"
    def test_flags = "--disable-verification-cache"
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined"
    def compilerpath = ""
    def configargs = ""
    if (prefixpath == "")
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

def buildJob(compiler, flags, env4make, image, prefixpath="/opt/rocm", cmd = ""){

        env.HSA_ENABLE_SDMA=0 
        checkout scm
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerArgs = "--build-arg PREFIX=${prefixpath} "
        if(prefixpath == "")
        {
            dockerArgs = ""
        }
        def retimage
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
                    cmake_build(compiler, flags, env4make, prefixpath)
                }else{
                    sh cmd
                }
            }
        }
        return retimage
}

def buildHipClangJob(compiler, flags, env4make, image, prefixpath="/opt/rocm", cmd = ""){

        env.HSA_ENABLE_SDMA=0 
        checkout scm
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerArgs = "--build-arg PREFIX=${prefixpath} -f hip-clang.docker "
        def retimage
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
        return retimage
}



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
                        buildJob('hcc', '-DCMAKE_BUILD_TYPE=release', "", image, "", cmd)
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
                        buildJob('hcc', '-DCMAKE_BUILD_TYPE=release', "", image, "", cmd)
                    }
                }

                stage('Hip Tidy') {
                    agent{ label rocmnode("rocmtest") }
                    environment{
                        cmd = "rm -rf build; mkdir build; cd build; CXX=/usr/local/bin/hcc cmake -DBUILD_DEV=On ..; make -j\$(nproc) -k analyze;"
                    }
                    steps{
                        buildJob('hcc', '-DCMAKE_BUILD_TYPE=release', "", image, "", cmd)
                    }
                }
            }
        }
        
        // Run quick fp32 tests
        stage("Fast full precision"){
            parallel{
               stage('Clang Debug') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', "", image, "")
                    }
                }

                stage('Clang Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', "", image, "")
                    }
                }

                stage('GCC Debug') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', "", image, "")
                    }
                }

                stage('GCC Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', "", image, "")
                    }
                }

                stage('Fiji GCC Debug') {
                    agent{ label rocmnode("fiji") }
                    steps{
                        buildJob('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', "", image, "")
                    }
                }

                stage('Hip Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('hcc', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', "", image + "rocm")
                    }
                }

                stage('Hip clang debug') {
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
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "", image+'-hip-clang', "/usr/local", cmd)
                    }
                }



                stage('gfx908 Hip debug') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', "", image + "rocm")
                    }
                }
            }
        }

        // Misc tests
        stage("Aux tests"){
            parallel{
                stage('Hip clang debug COMGR') {
                    agent{ label rocmnode("vega") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_USE_COMGR=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache .. 
                            CTEST_PARALLEL_LEVEL=4 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "MIOPEN_LOG_LEVEL=5",  image+'-hip-clang', "/usr/local", cmd)
                    }
                }

                stage('Hip Static Release') {
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
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "",  image+'-hip-clang', "/usr/local", cmd)
                    }
                }

                stage('Hip Normal Find Mode Release') {
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
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "",  image+'-hip-clang', "/usr/local", cmd)
                    }
                }

                stage('Hip Fast Find Mode Release') {
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
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "",  image+'-hip-clang', "/usr/local", cmd)
                    }
                }

                stage('Hip Release on /usr/local') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('hcc', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', "", image, "")
                    }
                }

            }
        }

        // Run fp16, bfp16, and int8 quick tests
        stage("Fast low precision"){
            parallel{
                stage('Half Hip Release') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', "", image + "rocm")
                    }
                }

                stage('Half GCC Debug') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('g++-5', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', "", image, "")
                    }
                }
    
                stage('Half GCC Release') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('g++-5', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', "", image, "")
                    }
                }

                stage('Int8 Hip Release') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', "", image + "rocm")
                    }
                }

                stage('Int8 GCC Debug') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('g++-5', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', "", image, "")
                    }
                }

                stage('Int8 GCC Release') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('g++-5', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', "", image, "")
                    }
                }

                stage('Bfloat16 Hip Release') {
                    agent{ label rocmnode("vega20") }   
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_BFLOAT16=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', "", image + "rocm")
                    }
                }

                stage('Bfloat16 gfx908 Hip Debug') {
                    agent{ label rocmnode("gfx908") }   
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', "", image + "rocm")
                    }
                }

                stage('Half gfx908 Hip Debug') {
                    agent{ label rocmnode("gfx908") }   
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', "", image + "rocm")
                    }
                }
            }
        }

        stage("Full tests I"){
            parallel{
                stage('Int8 Hip Release All') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', "", image + "rocm")
                    }
                }

                stage('Bfloat16 Hip Release All') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_BFLOAT16=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', "", image + "rocm")
                    }
                }

                stage('Bfloat16 gfx908 Hip Release All Subset') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_BFLOAT16=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', "", image + "rocm")
                    }
                }
            }
        }

        stage("Full tests II"){
            parallel{
                stage('GCC Release All') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('g++-5', '-DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', "", image, "")
                    }
                }

                stage('FP32 gfx908 Hip Release All subset') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', "MIOPEN_LOG_LEVEL=5", image + "rocm")
                    }
                }

                stage('Half gfx908 Hip Release All Subset') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_HALF=On -DMIOPEN_TEST_GFX908=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', "", image + "rocm")
                    }
                }
            }
        }

        stage("Full tests III"){
            parallel{
                stage('Half Hip Clang Release All') {
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
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "", image+'-hip-clang', "/usr/local", cmd)
                    }
                }

                stage('Hip Clang Release All') {
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
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "", image+'-hip-clang', "/usr/local", cmd)
                    }
                }
            }
        }


       // Run package building
        stage("Packages"){
            parallel {
                stage('GCC OpenCL Release package') {
                    agent{ label rocmnode("rocmtest") }
                    steps{
                        buildJob('g++-5', '-DCMAKE_BUILD_TYPE=release', "", image, "")
                    }
                }
                stage("HCC HIP Release package"){
                    agent{ label rocmnode("rocmtest") }
                    steps{
                        buildJob('hcc', '-DCMAKE_BUILD_TYPE=release', "", image + "rocm")
                    }
                }
            }
        }
    }    
}




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



def cmake_build(compiler, flags, prefixpath="/opt/rocm"){
    def workspace_dir = pwd()
    def vcache = "/var/jenkins/.cache/miopen/vcache"
    def archive = (flags == '-DCMAKE_BUILD_TYPE=release')
    def config_targets = "all" // "check doc MIOpenDriver"
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
        MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS=1 CTEST_PARALLEL_LEVEL=4 MIOPEN_VERIFY_CACHE_PATH=${vcache} MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 dumb-init make -j\$(nproc) ${config_targets}
    """
    echo cmd
    sh cmd
    // Only archive from master or develop
    if (archive == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}

def buildJob(compiler, flags, image, prefixpath="/opt/rocm", cmd = ""){

        env.HSA_ENABLE_SDMA=0
        checkout scm
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --user=root --privileged "
        def dockerArgs = "--build-arg PREFIX=${prefixpath} "
        if(prefixpath == "")
        {
            dockerArgs = ""
        }
        def retimage
        try {
            echo "build docker"
            retimage = docker.build("${image}", dockerArgs + '.')
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                }
            }
        } catch(Exception ex) {
            echo "exception ocurred"
            retimage = docker.build("${image}", dockerArgs + "--no-cache .")
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                }
            }
        }

        withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
            timeout(time: 5, unit: 'HOURS')
            {
                if(cmd == ""){
                    cmake_build(compiler, flags, prefixpath)
                }else{
                    echo "run shell command"
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
                        cmd = "rm -rf build; \
                                mkdir build; \
                                cd build; \
                                CXX='clang++-3.9' cmake -DBUILD_DEV=On ..; \
                                make -j\$(nproc) -k analyze;"
                    }
                    steps{
                        buildJob('clang++-3.9', '-DCMAKE_BUILD_TYPE=release', image, "", cmd)
                    }
                }

                stage('Clang Format') {
                    agent{ label rocmnode("rocmtest") }
                    environment{
                        cmd = "cd src; find . -iname \'*.h\' \
                                -o -iname \'*.hpp\' \
                                -o -iname \'*.cpp\' \
                                -o -iname \'*.h.in\' \
                                -o -iname \'*.hpp.in\' \
                                -o -iname \'*.cpp.in\' \
                                -o -iname \'*.cl\' \
                                | grep -v 'build/' \
                                | grep -v 'base64' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-3.9 -style=file {} | diff - {}\'"
                    }
                    steps{
                        buildJob('clang++-3.9', '-DCMAKE_BUILD_TYPE=release', image, "", cmd)
                    }
                }

                stage('Hip Tidy') {
                    agent{ label rocmnode("rocmtest") }
                    environment{
                        cmd = "rm -rf build; \
                                mkdir build; \
                                cd build; \
                                CXX=/usr/local/bin/hcc cmake -DBUILD_DEV=On ..; \
                                make -j\$(nproc) -k analyze;"
                    }
                    steps{
                        buildJob('hcc', '-DCMAKE_BUILD_TYPE=release', image, "", cmd)
                    }
                }
            }
        }



    }
}


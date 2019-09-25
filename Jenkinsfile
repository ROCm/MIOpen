

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
    } else {
        node_name = name
    }
    return node_name
}



def cmake_build(compiler, flags){
    def workspace_dir = pwd()
    def vcache = "/var/jenkins/.cache/miopen/vcache"
    def archive = (flags == '-DCMAKE_BUILD_TYPE=release')
    def config_targets = "check doc MIOpenDriver"
    if (archive == true) {
        config_targets = "package"
    }
    def cmd = """
        echo \$HSA_ENABLE_SDMA
        mkdir -p $WINEPREFIX
        rm -rf build
        mkdir build
        cd build
        CXX=${compiler} CXXFLAGS='-Werror' cmake -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS="--disable-verification-cache" -DCMAKE_CXX_FLAGS_DEBUG='-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined' ${flags} .. 
        CTEST_PARALLEL_LEVEL=4 MIOPEN_VERIFY_CACHE_PATH=${vcache} MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 dumb-init make -j32 ${config_targets}
    """
    echo cmd
    sh cmd
    // Only archive from master or develop
    if (archive == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}

def buildJob(compiler, flags, image, cmd = ""){

        env.HSA_ENABLE_SDMA=0 
        // env.HSA_ENABLE_INTERRUPT=0
        env.WINEPREFIX="/jenkins/.wine"
        checkout scm
        def retimage
        try {
            retimage = docker.build("${image}", "--build-arg PREFIX=/usr/local .")
            withDockerContainer(image: image, args: '--device=/dev/kfd --device=/dev/dri --group-add video') {
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'PATH="/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                }
            }
        } catch(Exception ex) {
            retimage = docker.build("${image}", "--build-arg PREFIX=/usr/local --no-cache .")
            withDockerContainer(image: image, args: '--device=/dev/kfd --device=/dev/dri --group-add video') {
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'PATH="/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                }
            }
        }

        withDockerContainer(image: image, args: '--device=/dev/kfd --device=/dev/dri --group-add video -v=/var/jenkins/:/var/jenkins') {
            timeout(time: 4, unit: 'HOURS')
            {
                if(cmd == ""){
                    cmake_build(compiler, flags)
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
                        cmd = "rm -rf build; mkdir build; cd build; CXX='clang++-3.8' cmake -DBUILD_DEV=On ..; make -j8 -k analyze;"
                    }
                    steps{
                        buildJob('hcc', '-DCMAKE_BUILD_TYPE=release', 'miopen', cmd)
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
                        buildJob('hcc', '-DCMAKE_BUILD_TYPE=release', 'miopen', cmd)
                    }
                }

                stage('Hip Tidy') {
                    agent{ label rocmnode("rocmtest") }
                    environment{
                        cmd = "rm -rf build; mkdir build; cd build; CXX='hcc' cmake -DBUILD_DEV=On ..; make -j8 -k analyze;"
                    }
                    steps{
                        buildJob('hcc', '-DCMAKE_BUILD_TYPE=release', 'miopen', cmd)
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
                        buildJob('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', image)
                    }
                }

                stage('Clang Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }

                stage('GCC Debug') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', image)
                    }
                }

                stage('GCC Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }

                stage('Fiji GCC Debug') {
                    agent{ label rocmnode("fiji") }
                    steps{
                        buildJob('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', image)
                    }
                }

                stage('Hip Release') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('hcc', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image)
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
                        buildJob('hcc', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }

                stage('Half GCC Debug') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('g++-5', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', image)
                    }
                }
    
                stage('Half GCC Release') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('g++-5', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }

                stage('Int8 Hip Release') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }

                stage('Int8 GCC Debug') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('g++-5', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug', image)
                    }
                }

                stage('Int8 GCC Release') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('g++-5', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }

                stage('Bfloat16 Hip Release') {
                    agent{ label rocmnode("vega20") }   
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_BFLOAT16=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }
            }
        }

        stage("Full short tests"){
            parallel{
                stage('Int8 Hip Release All') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }

                stage('Bfloat16 Hip Release All') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_BFLOAT16=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }
            }
        }

        stage("Full long tests"){
            parallel{
                stage('GCC Release All') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('g++-5', '-DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }

                stage('Hip Release All') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildJob('hcc', '-DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', image)
                    }
                }

                stage('Half Hip Release All') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildJob('hcc', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release', image)
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
                        buildJob('g++-5', '-DCMAKE_BUILD_TYPE=release', image)
                    }
                }
                stage("HCC HIP Release package"){
                    agent{ label rocmnode("rocmtest") }
                    steps{
                        buildJob('hcc', '-DCMAKE_BUILD_TYPE=release', image)
                    }
                }
            }
        }
    }    
}


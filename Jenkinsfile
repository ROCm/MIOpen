
def rocmtestnode(variant, name, body) {
    def image = 'miopen'
    def cmake_build = { compiler, flags ->
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
            CXX=${compiler} CXXFLAGS='-Werror' cmake -DMIOPEN_GPU_SYNC=On -DCMAKE_CXX_FLAGS_DEBUG='-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined' ${flags} .. 
            CTEST_PARALLEL_LEVEL=4 MIOPEN_VERIFY_CACHE_PATH=${vcache} MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 dumb-init make -j32 ${config_targets}
        """
        echo cmd
        sh cmd
        // Only archive from master or develop
        if (archive == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
            archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
        }
    }
    node(name) {
        stage("checkout ${variant}") {
            // env.HCC_SERIALIZE_KERNEL=3
            // env.HCC_SERIALIZE_COPY=3
            env.HSA_ENABLE_SDMA=0 
            // env.HSA_ENABLE_INTERRUPT=0
            env.WINEPREFIX="/jenkins/.wine"
            checkout scm
        }
        stage("image ${variant}") {
            try {
                docker.build("${image}", "--build-arg PREFIX=/usr/local .")
                withDockerContainer(image: image, args: '--device=/dev/kfd --device=/dev/dri --group-add video') {
                    sh 'PATH="/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                }
            } catch(Exception ex) {
                docker.build("${image}", "--build-arg PREFIX=/usr/local --no-cache .")
                withDockerContainer(image: image, args: '--device=/dev/kfd --device=/dev/dri --group-add video') {
                    sh 'PATH="/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                }
            }
        }
        withDockerContainer(image: image, args: '--device=/dev/kfd --device=/dev/dri --group-add video -v=/var/jenkins/:/var/jenkins') {
            timeout(time: 3, unit: 'HOURS') {
                body(cmake_build)
            }
        }
    }
}
@NonCPS
def rocmtest(m) {
    def builders = [:]
    for(e in m) {
        def label = e.key;
        def action = e.value;
        builders[label] = {
            action(label)
        }
    }
    parallel builders
}

@NonCPS
def rocmnode(name, body) {
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
    return { label ->
        rocmtestnode(label, node_name, body)
    }
}

@NonCPS
def rocmnode(body) {
    rocmnode('rocmtest', body)
}

// Static checks
rocmtest opencl_tidy: rocmnode('rocmtest') { cmake_build ->
    stage('Clang Tidy') {
        sh '''
            rm -rf build
            mkdir build
            cd build
            CXX='clang++-3.8' cmake -DBUILD_DEV=On .. 
            make -j8 -k analyze
        '''
    }
}, format: rocmnode('rocmtest') { cmake_build ->
    stage('Clang Format') {
        sh '''
            find . -iname \'*.h\' \
                -o -iname \'*.hpp\' \
                -o -iname \'*.cpp\' \
                -o -iname \'*.h.in\' \
                -o -iname \'*.hpp.in\' \
                -o -iname \'*.cpp.in\' \
                -o -iname \'*.cl\' \
            | grep -v 'build/' \
            | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-3.8 -style=file {} | diff - {}\'
        '''
    }
}, hip_tidy: rocmnode('rocmtest') { cmake_build ->
    stage('Hip Tidy') {
        sh '''
            rm -rf build
            mkdir build
            cd build
            CXX='hcc' cmake -DBUILD_DEV=On .. 
            make -j8 -k analyze
        '''
    }
}

// Quick tests
rocmtest opencl: rocmnode('vega') { cmake_build ->
    stage('Clang Debug') {
        cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('Clang Release') {
        cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
    stage('GCC Debug') {
        cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('GCC Release') {
        cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}, fiji: rocmnode('fiji') { cmake_build ->
    stage('Fiji GCC Debug') {
        cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
}, hip: rocmnode('vega') { cmake_build ->
    // stage('Hip Debug') {
    //     cmake_build('hcc', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_USE_ROCBLAS=ON')
    // }
    stage('Hip Release') {
        cmake_build('hcc', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}, half_hip: rocmnode('vega20') { cmake_build ->
    stage('Half Hip Release') {
        cmake_build('hcc', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}, half_opencl: rocmnode('vega20') { cmake_build ->
    stage('Half GCC Debug') {
        cmake_build('g++-5', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('Half GCC Release') {
        cmake_build('g++-5', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
// }, windows: rocmnode('fiji') { cmake_build ->
//     stage('Windows Release') {
//         cmake_build('x86_64-w64-mingw32-g++', '-DBUILD_DEV=On -DCMAKE_TOOLCHAIN_FILE=/usr/local/x86_64-w64-mingw32/cmake/toolchain.cmake -DCMAKE_BUILD_TYPE=release')
//     }
}, int8_hip: rocmnode('vega20') { cmake_build ->
    stage('Int8 Hip Release') {
        cmake_build('hcc', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}, int8_opencl: rocmnode('vega20') { cmake_build ->
    stage('Int8 GCC Debug') {
        cmake_build('g++-5', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('Int8 GCC Release') {
        cmake_build('g++-5', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}

// All tests
rocmtest opencl_all: rocmnode('vega') { cmake_build ->
    stage('GCC Release All') {
        cmake_build('g++-5', '-DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
    }
}, hip_all: rocmnode('vega') { cmake_build ->
    stage('Hip Release All') {
        cmake_build('hcc', '-DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
    }
}, half_hip_all: rocmnode('vega20') { cmake_build ->
    stage('Half Hip Release All') {
        cmake_build('hcc', '-DMIOPEN_TEST_HALF=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
    }
}, int8_hip_all: rocmnode('vega20') { cmake_build ->
    stage('Int8 Hip Release All') {
        cmake_build('hcc', '-DMIOPEN_TEST_INT8=On -DBUILD_DEV=On -DMIOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
    }
}



// Package stage
rocmtest opencl_package: rocmnode('rocmtest') { cmake_build ->
    stage('GCC Release Package') {
        cmake_build('g++-5', '-DCMAKE_BUILD_TYPE=release')
    }
}, hip_package: rocmnode('rocmtest') { cmake_build ->
    stage('Hip Release Package') {
        cmake_build('hcc', '-DCMAKE_BUILD_TYPE=release')
    }
}

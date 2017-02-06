// parallel opencl: {
//     node ('rocmtest') {
//         stage('OCL Checkout') {
//             checkout scm
//         }
//         withDockerContainer(image: 'rocm-opencl:1.4', args: '--device=/dev/kfd') {
//             timeout(time: 1, unit: 'HOURS') {
//                 stage('Clang Tidy') {
//                     sh '''
//                         rm -rf build
//                         mkdir build
//                         cd build
//                         CXX='clang++-3.8' cmake -DBUILD_DEV=On .. 
//                         make tidy 2>&1 | tee tidy_out
//                         ! grep -q "warning:" tidy_out
//                     '''
//                 }
//                 stage('Clang Debug') {
//                     cmake_build(compiler: 'clang++-3.8', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
//                 }
//                 stage('Clang Release') {
//                     cmake_build(compiler: 'clang++-3.8', flags: '-DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
//                 }
//                 stage('GCC Debug') {
//                     cmake_build(compiler: 'g++-4.8', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
//                 }
//                 stage('GCC Release') {
//                     cmake_build(compiler: 'g++-4.8', flags: '-DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
//                 }
//             }
//         }
//     }
// }, hip: {
//     node ('rocmtest') {
//         stage('HIP Checkout') {
//             checkout scm
//         }
//         withDockerContainer(image: 'aoc2:latest', args: '--device=/dev/kfd') {
//             timeout(time: 1, unit: 'HOURS') {
//                 stage('Hip Debug') {
//                     cmake_build(compiler: 'hcc', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
//                 }
//                 stage('Hip Release') {
//                     cmake_build(compiler: 'hcc', flags: '-DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
//                 }
//             }
//         }
//     }
// }

// node ('rocmtest') {
//     stage('OCL Checkout') {
//         checkout scm
//     }
//     cmake_step(image: 'rocm-opencl:1.4', stage: 'Clang Debug', compiler: 'clang++-3.8', flags: '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
// }
parallel opencl: {
    rocmtest('rocm-opencl:1.4') { cmake_build ->
        // stage('Clang Tidy') {
        //     sh '''
        //         rm -rf build
        //         mkdir build
        //         cd build
        //         CXX='clang++-3.8' cmake -DBUILD_DEV=On .. 
        //         make tidy 2>&1 | tee tidy_out
        //         ! grep -q "warning:" tidy_out
        //     '''
        // }
        stage('Clang Debug') {
            cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
        }
        stage('Clang Release') {
            cmake_build('clang++-3.8', '-DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
        }
        stage('GCC Debug') {
            cmake_build('g++-4.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
        }
        stage('GCC Release') {
            cmake_build('g++-4.8', '-DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
        }
    }
}, hip: {
    rocmtest('aoc2:latest') { cmake_build ->
        stage('Hip Debug') {
            cmake_build('hcc', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
        }
        stage('Hip Release') {
            cmake_build('hcc', '-DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
        }
    }
}

def rocmtest(image, body) {
    def cmake_build = { compiler, flags ->
        def cmd = """
            rm -rf build
            mkdir build
            cd build
            CXX=${compiler} CXXFLAGS='-Werror' cmake ${flags} .. 
            CTEST_PARALLEL_LEVEL=32 dumb-init make -j32 check
        """
        echo cmd
        sh cmd
    }
    node('rocmtest') {
        checkout scm
        withDockerContainer(image: image, args: '--device=/dev/kfd') {
            timeout(time: 1, unit: 'HOURS') {
                body(cmake_build)
            }
        }
    }
}

// def cmake_step(stage, compiler, flags) {
//     stage(stage) {
//         sh '''
//             rm -rf build
//             mkdir build
//             cd build
//             CXX=${compiler} CXXFLAGS='-Werror' cmake ${flags} .. 
//             CTEST_PARALLEL_LEVEL=32 dumb-init make -j32 check
//         '''
//     }
// }

// def cmake_build(compiler, flags) {
//     sh '''
//         rm -rf build
//         mkdir build
//         cd build
//         CXX=${compiler} CXXFLAGS='-Werror' cmake ${flags} .. 
//         CTEST_PARALLEL_LEVEL=32 dumb-init make -j32 check
//     '''
// }

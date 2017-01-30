parallel opencl: {
    node ('rocmtest') {
        stage('Checkout') {
            env.CXXFLAGS = "-Werror"
            env.CTEST_PARALLEL_LEVEL = "32"
            checkout scm
        }
        stage('Clang Tidy') {
            sh '''
                rm -rf build
                mkdir build
                cd build
                CXX='clang++-3.8' cmake -DBUILD_DEV=On .. 
                make tidy 2>&1 | tee tidy_out
                ! grep -q "warning:" tidy_out
            '''
        }
        stage('Clang Debug') {
            sh '''
                rm -rf build
                mkdir build
                cd build
                CXX='clang++-3.8' cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug .. 
                make -j32 check
            '''
        }
        stage('Clang Release') {
            sh '''
                rm -rf build
                mkdir build
                cd build
                CXX='clang++-3.8' cmake -DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release .. 
                make -j32 check
            '''
        }
        stage('GCC Debug') {
            sh '''
                rm -rf build
                mkdir build
                cd build
                CXX='g++-4.8' cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug .. 
                make -j32 check
            '''
        }
        stage('GCC Release') {
            sh '''
                rm -rf build
                mkdir build
                cd build
                CXX='g++-4.8' cmake -DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release .. 
                make -j32 check
            '''
        }
    }
}, hip: {
    node ('rocmtest') {
        stage('Checkout') {
            env.CXXFLAGS = "-Werror"
            env.CTEST_PARALLEL_LEVEL = "32"
            checkout scm
        }
        withDockerContainer(image: 'rocm-aoc2:tot', args: '--device=/dev/kfd') {

            stage('Hip Debug') {
                sh '''
                    rm -rf build
                    mkdir build
                    cd build
                    CXX='hcc' cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug .. 
                    make -j32 check
                '''
            }
            stage('Hip Release') {
                sh '''
                    rm -rf build
                    mkdir build
                    cd build
                    CXX='hcc' cmake -DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release .. 
                    make -j32 check
                '''
            }
        }
    }
}

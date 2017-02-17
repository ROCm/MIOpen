parallel opencl: {
    rocmtest('tinygemm:1.4') { cmake_build ->
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
        // stage('Hip Debug') {
        //     cmake_build('hcc', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
        // }
        stage('Hip Release') {
            cmake_build('hcc', '-DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release')
        }
    }
}

def rocmtest(image, body) {
    def cmake_build = { compiler, flags ->
        def cmd = """
            echo \$HSA_ENABLE_SDMA
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
        stage("checkout ${image}") {
            env.HCC_SERIALIZE_KERNEL=3
            env.HCC_SERIALIZE_COPY=3
            env.HSA_ENABLE_SDMA=0
            env.HSA_ENABLE_INTERRUPT=0
            checkout scm
        }
        withDockerContainer(image: image, args: '--device=/dev/kfd') {
            timeout(time: 1, unit: 'HOURS') {
                body(cmake_build)
            }
        }
    }
}


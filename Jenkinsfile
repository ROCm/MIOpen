node ('rocmtest10'){
    stage 'Checkout'
    env.CXXFLAGS = "-Werror"
    env.CTEST_PARALLEL_LEVEL = "4"
    checkout scm
    stage 'Clang Tidy'
    sh '''
        rm -rf build
        mkdir build
        cd build
        CXX='clang++-3.8' cmake -DBUILD_DEV=On .. 
        make tidy 2>&1 | tee tidy_out
        ! grep -q "warning:" tidy_out
    '''
    stage 'Clang Debug'
    sh '''
        rm -rf build
        mkdir build
        cd build
        CXX='clang++-3.8' cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug .. 
        make check
    '''
    stage 'Clang Release'
    sh '''
        rm -rf build
        mkdir build
        cd build
        CXX='clang++-3.8' cmake -DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release .. 
        make check
    '''
    stage 'GCC Debug'
    sh '''
        rm -rf build
        mkdir build
        cd build
        CXX='g++-4.8' cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug .. 
        make check
    '''
    stage 'GCC Release'
    sh '''
        rm -rf build
        mkdir build
        cd build
        CXX='g++-4.8' cmake -DBUILD_DEV=On -DMLOPEN_TEST_ALL=On -DCMAKE_BUILD_TYPE=release .. 
        make check
    '''
 }

node ('fglrx1'){
    stage 'Checkout'
    env.CXXFLAGS = "-Werror"
    checkout scm
    stage 'Clang Tidy'
    sh '''
        rm -rf build
        mkdir build
        cd build
        CXX='clang++-3.8' cmake -DBUILD_DEV=On .. 
        make tidy
    '''
    stage 'Clang Debug'
    sh '''
        rm -rf build
        mkdir build
        cd build
        CXX='clang++-3.8' cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug .. 
        make check
        make tidy
    '''
    stage 'Clang Release'
    sh '''
        rm -rf build
        mkdir build
        cd build
        CXX='clang++-3.8' cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release .. 
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
        CXX='g++-4.8' cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release .. 
        make check
    '''
 }

node ('fglrx1'){
    stage 'Checkout'
    env.CXXFLAGS = "-Werror"
    checkout scm
    stage 'Debug build'
    sh '''
        rm -rf build
        mkdir build
        cd build
        CXX='clang++-3.8' cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug .. 
        make check
        make tidy
    '''
    stage 'Release build'
    sh '''
        rm -rf build
        mkdir build
        cd build
        CXX='clang++-3.8' cmake -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release .. 
        make check
        make tidy
    '''


 }

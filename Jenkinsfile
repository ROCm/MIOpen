node ('fglrx1'){
  stage 'Build and Test'
  env.CXX = "clang++-3.8"
  env.CXXFLAGS = "-Werror"
  checkout scm
  sh '''
    rm -rf build
    mkdir build
    cd build
    cmake .. 
    make check
    make tidy
    '''
 }

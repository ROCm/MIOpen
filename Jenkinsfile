node ('fglrx1'){
  stage 'Build and Test'
  env.CXX = "clang++-3.8"
  env.CXXFLAGS = "-Werror"
  checkout scm
  sh '''
    mkdir build
    cd build
    cmake .. 
    make check
    make tidy
    '''
 }

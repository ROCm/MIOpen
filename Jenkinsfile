def rocmnode(name) {
    return 'rocmtest && miopen && ' + name
}

def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME"
        lsb_release -sd
        uname -r
        cat /sys/module/amdgpu/version
        ls /opt/ -la
    """
}

//default
// CXX=/opt/rocm/llvm/bin/clang++ CXXFLAGS='-Werror' cmake -DMIOPEN_GPU_SYNC=Off -DCMAKE_PREFIX_PATH=/usr/local -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release ..
//
def cmake_build(Map conf=[:]){

    def compiler = conf.get("compiler","/opt/rocm/llvm/bin/clang++")
    def config_targets = conf.get("config_targets","check")
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined -Wno-option-ignored " + conf.get("extradebugflags", "")
    def build_envs = "CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 " + conf.get("build_env","")
    def prefixpath = conf.get("prefixpath","/usr/local")
    def mlir_args = " -DMIOPEN_USE_MLIR=" + conf.get("mlir_build", "ON")
    def setup_args = mlir_args + " -DMIOPEN_GPU_SYNC=Off " + conf.get("setup_flags","")

    if (prefixpath != "/usr/local"){
        setup_args = setup_args + " -DCMAKE_PREFIX_PATH=${prefixpath} "
    }

    def build_type_debug = (conf.get("build_type",'release') == 'debug')

    //cmake_env can overwrite default CXX variables.
    def cmake_envs = "CXX=${compiler} CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")

    def package_build = (conf.get("package_build","") == "true")

    if (package_build == true) {
        config_targets = "package"
    }

    if(conf.get("build_install","") == "true")
    {
        config_targets = 'install ' + config_targets
        setup_args = ' -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install' + setup_args
    } else{
        setup_args = ' -DBUILD_DEV=On' + setup_args
    }

    // test_flags = ctest -> MIopen flags
    def test_flags = conf.get("test_flags","")

    if (conf.get("vcache_enable","") == "true"){
        def vcache = conf.get(vcache_path,"/var/jenkins/.cache/miopen/vcache")
        build_envs = " MIOPEN_VERIFY_CACHE_PATH='${vcache}' " + build_envs
    } else{
        test_flags = " --disable-verification-cache " + test_flags
    }

    if(conf.get("codecov", false)){ //Need
        setup_args = " -DCMAKE_BUILD_TYPE=debug -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags} -fprofile-arcs -ftest-coverage' -DCODECOV_TEST=On " + setup_args
    }else if(build_type_debug){
        setup_args = " -DCMAKE_BUILD_TYPE=debug -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'" + setup_args
    }else{
        setup_args = " -DCMAKE_BUILD_TYPE=release" + setup_args
    }

    if(test_flags != ""){
       setup_args = "-DMIOPEN_TEST_FLAGS='${test_flags}'" + setup_args
    }

    if(conf.containsKey("find_mode"))
    {
        def fmode = conf.get("find_mode", "")
        setup_args = " -DMIOPEN_DEFAULT_FIND_MODE=${fmode} " + setup_args
    }
    if(env.CCACHE_HOST)
    {
        setup_args = " -DCMAKE_CXX_COMPILER_LAUNCHER='ccache' -DCMAKE_C_COMPILER_LAUNCHER='ccache' " + setup_args
    }

    def pre_setup_cmd = """
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            rm -rf build
            mkdir build
            rm -rf install
            mkdir install
            rm -f src/kernels/*.ufdb.txt
            rm -f src/kernels/miopen*.udb
            cd build
        """
    def setup_cmd = conf.get("setup_cmd", "${cmake_envs} cmake ${setup_args}   .. ")
    // WORKAROUND_SWDEV_290754
    // It seems like this W/A is not required since 4.5.
    def build_cmd = conf.get("build_cmd", "LLVM_PATH=/opt/rocm/llvm ${build_envs} dumb-init make -j\$(nproc) ${config_targets}")
    def execute_cmd = conf.get("execute_cmd", "")

    def cmd = conf.get("cmd", """
            ${pre_setup_cmd}
            ${setup_cmd}
            ${build_cmd}
            ${execute_cmd}
        """)

    echo cmd
    sh cmd

    // Only archive from master or develop
    if (package_build == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}
def getDockerImageName(prefixpath)
{
    def branch =  sh(script: "echo ${scm.branches[0].name} | sed 's/[^a-zA-Z0-9]/_/g' ", returnStdout: true).trim()
    def image = "${env.MIOPEN_IMAGE_URL}:miopen_ci_${branch}"
    if(prefixpath == "/usr/local")
    {
        image = image + "_usr"
    }
    else if(prefixpath == "/opt/rocm")
    {
        image = image + "_opt"
    }
    else
    {
        error "Unknown prefixpath: ${prefixpath}"
    }
    return image

}
def getDockerImage(Map conf=[:])
{
    env.DOCKER_BUILDKIT=1
    def prefixpath = conf.get("prefixpath", "/usr/local") // one image for each prefix 1: /usr/local 2:/opt/rocm
    def gpu_arch = conf.get("gpu_arch", "gfx900;gfx906") // prebuilt dockers should have all the architectures enabled so one image can be used for all stages
    def miotensile_version = conf.get("miotensile_version", "default") // deprecated
    def target_id = conf.get("target_id", "OFF") // deprecated
    def mlir_build = conf.get("mlir_build", "ON") // always ON
    def build_fin = conf.get("build_fin", "OFF") // forcing this to be always on means fewer docker containers since this ensures all dependencies are present in the docker image
    def no_cache = conf.get("no_cache", false)
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg MIOTENSILE_VER='${miotensile_version}' --build-arg USE_TARGETID='${target_id}' --build-arg USE_MLIR='${mlir_build}' --build-arg USE_FIN='${build_fin}' "
    if(env.CCACHE_HOST)
    {
        def check_host = sh(script:"""(printf "PING\r\n";) | nc -N ${env.CCACHE_HOST} 6379 """, returnStdout: true).trim()
        if(check_host == "+PONG")
        {
            echo "FOUND CCACHE SERVER: ${CCACHE_HOST}"
        }
        else 
        {
            echo "CCACHE SERVER: ${CCACHE_HOST} NOT FOUND, got ${check_host} response"
        }
        dockerArgs = dockerArgs + " --build-arg CCACHE_SECONDARY_STORAGE='redis://${env.CCACHE_HOST}' --build-arg COMPILER_LAUNCHER='ccache' "
        env.CCACHE_DIR = """/tmp/ccache_store"""
        env.CCACHE_SECONDARY_STORAGE="""redis://${env.CCACHE_HOST}"""
    }
    if(no_cache)
    {
        dockerArgs = dockerArgs + " --no-cache "
    }
    echo "Docker Args: ${dockerArgs}"
    def image = getDockerImageName(prefixpath)
    //Check if image exists 
    def retimage
    try 
    {
        echo "Pulling down image: ${image}"
        retimage = docker.image("${image}")
        retimage.pull()
    }
    catch(Exception ex)
    {
        error "Unable to locate image: ${image}"
    }
    return [retimage, image]
}

def buildHipClangJob(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        env.CODECOV_TOKEN="aec031be-7673-43b5-9840-d8fb71a2354e"
        env.DOCKER_BUILDKIT=1
        def image
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1"
        }

        def variant = env.STAGE_NAME

        def codecov = conf.get("codecov", false)
        def needs_gpu = conf.get("needs_gpu", true)

        def retimage
        gitStatusWrapper(credentialsId: "${env.status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'MIOpen') {
            try {
                (retimage, image) = getDockerImage(conf)
                if (needs_gpu) {
                    withDockerContainer(image: image, args: dockerOpts) {
                        timeout(time: 5, unit: 'MINUTES')
                        {
                            sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                        }
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }
            catch(Exception ex) {
                conf.put("no_cache", true)
                (retimage, image) = getDockerImage(conf)
                retimage = docker.build("${image}", dockerArgs + "--no-cache .")
                if (needs_gpu) {
                    withDockerContainer(image: image, args: dockerOpts) {
                        timeout(time: 5, unit: 'MINUTES')
                        {
                            sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                        }
                    }
                }
            }

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 5, unit: 'HOURS')
                {
                    cmake_build(conf)

                    if (codecov) {
                        sh '''
                            cd build
                            lcov --directory . --capture --output-file $(pwd)/coverage.info
                            lcov --remove $(pwd)/coverage.info '/usr/*' --output-file $(pwd)/coverage.info
                            lcov --list $(pwd)/coverage.info
                            curl -s https://codecov.io/bash | bash
                            echo "Uploaded"
                        '''
                    }
                }
            }
        }
        return retimage
}

def reboot(){
    build job: 'reboot-slaves', propagate: false , parameters: [string(name: 'server', value: "${env.NODE_NAME}"),]
}

def buildHipClangJobAndReboot(Map conf=[:]){
    try{
        buildHipClangJob(conf)
    }
    catch(e){
        echo "throwing error exception for the stage"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (conf.get("needs_gpu", true)) {
            reboot()
        }
    }
}

def CheckDeserializePerfDb(Map conf=[:]){
    def pdb_image = buildHipClangJob(conf)
    pdb_image.inside(){
        sh "MIOPEN_LOG_LEVEL=4 LD_LIBRARY_PATH='install/lib:/opt/rocm/lib/' install/bin/fin -i fin/tests/pdb_check_all.json -o pdb_deserialize_error.json"
        archiveArtifacts "pdb_deserialize_error.json"
        sh "grep clear pdb_deserialize_error.json"
        def has_error = sh (
            script: "echo \$?",
            returnStdout: true
        ).trim()
        assert has_error.toInteger() == 0
    }
}

def buildDocker(install_prefix)
{
    env.DOCKER_BUILDKIT=1
    checkout scm
    def image_name = getDockerImageName(install_prefix)
    echo "Building Docker for ${image_name}"
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${install_prefix} --build-arg GPU_ARCH='gfx900;gfx906;gfx908;gfx90a;gfx90a:xnack-;gfx1030' --build-arg MIOTENSILE_VER='default' --build-arg USE_TARGETID='OFF' --build-arg USE_MLIR='ON' --build-arg USE_FIN='ON' "
    if(env.CCACHE_HOST)
    {
        def check_host = sh(script:"""(printf "PING\\r\\n";) | nc  -N ${env.CCACHE_HOST} 6379 """, returnStdout: true).trim()
        if(check_host == "+PONG")
        {
            echo "FOUND CCACHE SERVER: ${CCACHE_HOST}"
        }
        else 
        {
            echo "CCACHE SERVER: ${CCACHE_HOST} NOT FOUND, got ${check_host} response"
        }
        dockerArgs = dockerArgs + " --build-arg CCACHE_SECONDARY_STORAGE='redis://${env.CCACHE_HOST}' --build-arg COMPILER_LAUNCHER='ccache' "
        env.CCACHE_DIR = """/tmp/ccache_store"""
        env.CCACHE_SECONDARY_STORAGE="""redis://${env.CCACHE_HOST}"""
    }

    echo "Build Args: ${dockerArgs}"
    try 
    {
        echo "Checking for image: ${image_name}"
        sh "docker manifest inspect --insecure ${image_name}"
        echo "Image: ${image_name} found!! Skipping building image"
    }
    catch(Exception ex)
    {
        echo "Unable to locate image: ${image_name}. Building image now"
        retimage = docker.build("${image_name}", dockerArgs + ' .')
        retimage.push()
    }
}


/// Stage name format:
/// [DataType] Backend[/Compiler] BuildType [TestSet] [Target]
///
/// The only mandatory elements are Backend and BuildType; others are optional.
///
/// DataType := { Fp16 | Bf16 | Int8 | Fp32 }
/// Backend := { Hip | OpenCL | HipNoGPU}
/// Compiler := { Clang* | GCC* }
///   * "Clang" is the default for the Hip backend, and implies hip-clang compiler.
///     For the OpenCL backend, "Clang" implies the system x86 compiler.
///   * "GCC" is the default for OpenCL backend.
///   * The default compiler is usually not specified.
/// BuildType := { Release* | Debug | Install } [ BuildTypeModifier ]
///   * BuildTypeModifier := { NOCOMGR | Embedded | Static | Normal-Find | Fast-Find
///                            MLIR | Tensile | Tensile-Latest | Package | ... }
/// TestSet := { All | Smoke* } [ Codecov ]
///   * "All" corresponds to "cmake -DMIOPEN_TEST_ALL=On".
///   * "Smoke" (-DMIOPEN_TEST_ALL=Off) is the default and usually not specified.
///   * "Codecov" is optional code coverage analysis.
/// Target := { gfx908 | gfx90a | Vega20 | Vega10 | Vega* | gfx1030 } [ Xnack+ ]
///   * "Vega" (gfx906 or gfx900) is the default and usually not specified.


pipeline {
    agent none
    options {
        parallelsAlwaysFailFast()
        // disable stage-wise timeout due to long wait with queue (limited resources)
        // timeout(time: 90, unit:'MINUTES')
    }
    parameters {
        booleanParam(
            name: "BUILD_DOCKER",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "BUILD_STATIC_CHECKS",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "BUILD_SMOKE_FP32",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "BUILD_SMOKE_AUX1",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "BUILD_SMOKE_FP16_BF16_INT8",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "BUILD_FULL_TESTS1",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "BUILD_FULL_TESTS2",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "BUILD_PACKAGES",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "TARGET_NOGPU",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "TARGET_VEGA10",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "TARGET_VEGA20",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "TARGET_GFX908",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "TARGET_GFX90A",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "TARGET_NAVI21",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "DATATYPE_NA",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "DATATYPE_FP32",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "DATATYPE_FP16",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "DATATYPE_BF16",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "DATATYPE_INT8",
            defaultValue: true,
            description: "")
    }

    environment{
        extra_log_env   = " MIOPEN_LOG_LEVEL=5 "
        Fp16_flags      = " -DMIOPEN_TEST_HALF=On"
        Bf16_flags      = " -DMIOPEN_TEST_BFLOAT16=On"
        Int8_flags      = " -DMIOPEN_TEST_INT8=On"
        Full_test       = " -DMIOPEN_TEST_ALL=On"
        Smoke_targets = "check doc MIOpenDriver"
        NOCOMGR_flags   = " -DMIOPEN_USE_COMGR=Off"
    }
    stages{
        stage("Build Docker"){
            when {
                expression {params.BUILD_DOCKER && params.TARGET_NOGPU}
            }
            parallel{
                stage('Docker /opt/rocm'){
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildDocker('/opt/rocm')
                    }
                }
                stage('Docker /usr/local'){
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildDocker('/usr/local')
                    }
                }
            }
        }
        stage("Static checks") {
            when {
                expression { params.BUILD_STATIC_CHECKS && params.TARGET_NOGPU && params.DATATYPE_NA }
            }
            parallel{
                stage('Hip Tidy') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        setup_cmd = "CXX='/opt/rocm/llvm/bin/clang++' cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On .. "
                        build_cmd = "make -j\$(nproc) -k analyze"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: setup_cmd, build_cmd: build_cmd, needs_gpu:false)
                    }
                }
                stage('OpenCL Tidy') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        setup_cmd = "cmake -DMIOPEN_BACKEND=OpenCL -DBUILD_DEV=On .."
                        build_cmd = "make -j\$(nproc) -k analyze"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: setup_cmd, build_cmd: build_cmd, needs_gpu:false)
                    }
                }
                stage('Clang Format') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        execute_cmd = "find .. -iname \'*.h\' \
                                -o -iname \'*.hpp\' \
                                -o -iname \'*.cpp\' \
                                -o -iname \'*.h.in\' \
                                -o -iname \'*.hpp.in\' \
                                -o -iname \'*.cpp.in\' \
                                -o -iname \'*.cl\' \
                                | grep -v -E '(build/)|(install/)' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-10 -style=file {} | diff - {}\'"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd, needs_gpu:false)
                    }
                }
                stage('Tuna Fin Build Test') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                      setup_cmd = "CXX='/opt/rocm/llvm/bin/clang++' cmake -DCMAKE_BUILD_TYPE=DEBUG -DMIOPEN_BACKEND=HIPNOGPU -DBUILD_SHARED_LIBS=Off -DMIOPEN_INSTALL_CXX_HEADERS=On -DMIOPEN_ENABLE_FIN=ON .. "
                      build_cmd = "make -j\$(nproc) "
                    }
                    steps{
                      buildHipClangJobAndReboot(setup_cmd: setup_cmd, execute_cmd: "", build_cmd: build_cmd, build_fin: "ON", needs_gpu:false)
                  }
                }
                stage('Perf DB Deserialize Test') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        fin_flags = "-DCMAKE_BUILD_TYPE=DEBUG -DMIOPEN_BACKEND=HIPNOGPU -DBUILD_SHARED_LIBS=Off -DMIOPEN_INSTALL_CXX_HEADERS=On -DMIOPEN_ENABLE_FIN=ON"

                    }
                    steps{
                        CheckDeserializePerfDb(setup_flags: fin_flags, build_fin: "ON", config_targets: "MIOpenDriver", build_install: "true", needs_gpu:false)
                    }
                }
                stage('HipNoGPU Debug Build Test') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_NOGPU }
                    }
                    agent{ label rocmnode("nogpu") }
                    environment{
                        HipNoGPU_flags = "-DMIOPEN_BACKEND=HIPNOGPU -DMIOPEN_INSTALL_CXX_HEADERS=On"
                        build_cmd = "make -j\$(nproc)"
                    }
                    steps{
                        buildHipClangJob( build_type: 'debug', setup_flags: HipNoGPU_flags, build_cmd: build_cmd, needs_gpu:false)
                    }
                }
            }
        }
        stage("Smoke Fp32") {
            when {
                expression { params.BUILD_SMOKE_FP32 && params.DATATYPE_FP32 }
            }
            parallel{
               stage('Fp32 OpenCL Debug + Codecov') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 || params.TARGET_VEGA10 }
                    }
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', build_type: 'debug', config_targets: Smoke_targets, codecov: true)
                    }
                }
                stage('Fp32 OpenCL gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 }
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', config_targets: Smoke_targets, gpu_arch: "gfx908")
                    }
                }
                stage('Fp32 OpenCL gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', config_targets: Smoke_targets, gpu_arch: "gfx90a:xnack-")
                    }
                }
                stage('Fp32 Hip /opt/rocm') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 || params.TARGET_VEGA10 }
                    }
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(prefixpath: '/opt/rocm', config_targets: Smoke_targets)
                    }
                }
                stage('Fp32 Hip Debug') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 || params.TARGET_VEGA10 }
                    }
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(build_type: 'debug', config_targets: Smoke_targets)
                    }
                }
                stage('Fp32 OpenCL Debug gfx1030') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_NAVI21 }
                    }
                    agent{ label rocmnode("navi21") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', build_type: 'debug', config_targets: Smoke_targets, build_env: extra_log_env, gpu_arch: "gfx1030")
                    }
                }
                stage('Fp32 Hip Debug gfx908 /opt/rocm') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 }
                    }
                    agent{ label rocmnode("gfx908") }
                    environment{
                        gpu_arch = "gfx908"
                        prefixpath = "/opt/rocm"
                    }
                    steps{
                        buildHipClangJobAndReboot(prefixpath: prefixpath, build_type: 'debug', config_targets: Smoke_targets, gpu_arch: gpu_arch)
                    }
                }
                stage('Fp32 Hip Debug gfx90a /opt/rocm') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        gpu_arch = "gfx90a:xnack-"
                        prefixpath = "/opt/rocm"
                    }
                    steps{
                        buildHipClangJobAndReboot(prefixpath: prefixpath, build_type: 'debug', config_targets: Smoke_targets, gpu_arch: gpu_arch)
                    }
                }
            }
        }
        stage("Smoke Aux 1") {
            when {
                expression { params.BUILD_SMOKE_AUX1 && params.DATATYPE_FP32 }
            }
            parallel{
                stage('Fp32 Hip Debug NOCOMGR') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 || params.TARGET_VEGA10 }
                    }
                    agent{ label rocmnode("vega") }
                    environment{
                        // Can be removed altogether with when WORKAROUND_SWDEV_290754.
                        NOCOMGR_build_cmd = "CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( build_type: 'debug', setup_flags: NOCOMGR_flags, build_cmd: NOCOMGR_build_cmd, test_flags: ' --verbose ')
                    }
                }
                stage('Fp32 Hip Debug Embedded Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 }
                    }
                    agent{ label rocmnode("vega20") }
                    environment{
                        Embedded_flags = "-DMIOPEN_EMBED_DB='gfx906_60'"
                    }
                    steps{
                        buildHipClangJobAndReboot( build_type: 'debug', setup_flags: Embedded_flags, build_env: extra_log_env, test_flags: ' --verbose ')
                    }
                }
                stage('Fp32 Hip Static') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 || params.TARGET_VEGA10 }
                    }
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: "-DBUILD_SHARED_LIBS=Off", mlir_build: 'OFF')
                    }
                }
                stage('Fp32 Hip Normal-Find') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 || params.TARGET_VEGA10 }
                    }
                    agent{ label rocmnode("vega") }
                    environment{
                        config_targets = "test_conv2d"
                        execute_cmd = "MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot(config_targets: config_targets, execute_cmd: execute_cmd, find_mode: "Normal")
                    }
                }
                stage('Fp32 Hip Fast-Find') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 || params.TARGET_VEGA10 }
                    }
                    agent{ label rocmnode("vega") }
                    environment{
                        config_targets =   "test_conv2d"
                        execute_cmd = "MIOPEN_FIND_MODE=2 CTEST_PARALLEL_LEVEL=4  MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot( config_targets: config_targets, execute_cmd: execute_cmd)
                    }
                }
                stage('Fp32 Hip') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 || params.TARGET_VEGA10 }
                    }
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot()
                    }
                }
            }
        }
        stage("Smoke Fp16/Bf16/Int8") {
            when {
                expression { params.BUILD_SMOKE_FP16_BF16_INT8 }
            }
            parallel{
                stage('Fp16 OpenCL Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_FP16 }
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', setup_flags: Fp16_flags, config_targets: Smoke_targets)
                    }
                }
                stage('Int8 OpenCL Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_INT8 }
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', setup_flags: Int8_flags, config_targets: Smoke_targets)
                    }
                }
                stage('Fp16 Hip Vega20 /opt/rocm') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_FP16 }
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Fp16_flags, prefixpath: '/opt/rocm', config_targets: Smoke_targets)
                    }
                }
                stage('Bf16 Hip Vega20 /opt/rocm') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_BF16 }
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags, prefixpath: '/opt/rocm', config_targets: Smoke_targets)
                    }
                }
                stage('Fp16 Hip gfx908 /opt/rocm') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_FP16 }
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Fp16_flags, prefixpath: '/opt/rocm', config_targets: Smoke_targets, gpu_arch: "gfx908")
                    }
                }
                stage('Bf16 Hip gfx908 /opt/rocm') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_BF16 }
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags, prefixpath: '/opt/rocm', config_targets: Smoke_targets, gpu_arch: "gfx908")
                    }
                }
                stage('Fp16 Hip gfx90a /opt/rocm') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_FP16 }
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Fp16_flags, prefixpath: '/opt/rocm', config_targets: Smoke_targets, gpu_arch: "gfx90a:xnack-")
                    }
                }
                stage('Bf16 Hip gfx90a /opt/rocm') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_BF16 }
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags, prefixpath: '/opt/rocm', config_targets: Smoke_targets, gpu_arch: "gfx90a:xnack-")
                    }
                }
            }
        }
        stage("Full Tests I") {
            when {
                expression { params.BUILD_FULL_TESTS1 }
            }
            parallel{
                stage('Int8 HIP All Vega20 /opt/rocm') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_INT8 }
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Int8_flags + Full_test, prefixpath: '/opt/rocm')
                    }
                }
                stage('Fp32 OpenCL Install All') {
                    when {
                        beforeAgent true
                        expression { (params.TARGET_VEGA20 || params.TARGET_VEGA10) && params.DATATYPE_FP32 }
                    }
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', setup_flags: Full_test, build_install: "true")
                    }
                }
                stage('Bf16 Hip Install All gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_BF16 }
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: "true", gpu_arch: "gfx908")
                    }
                }
                stage('Bf16 Hip Install All gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_BF16 }
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: "true", gpu_arch: "gfx90a:xnack-")
                    }
                }
                stage('Fp32 OpenCL All gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_FP32 }
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', setup_flags: Full_test, gpu_arch: "gfx908")
                    }
                }
                stage('Fp32 OpenCL Install All gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_FP32 }
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', setup_flags: Full_test, build_install: "true", gpu_arch: "gfx90a:xnack-")
                    }
                }
                stage('Fp16 Hip All gfx1030') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_NAVI21 && params.DATATYPE_FP16 }
                    }
                    agent{ label rocmnode("navi21") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, gpu_arch: "gfx1030")
                    }
                }
            }
        }

        stage("Full Tests II") {
            when {
                expression { params.BUILD_FULL_TESTS2 }
            }
            environment{
                WORKAROUND_iGemm_936 = " MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0"
                // WORKAROUND_ISSUE_1148: "CTEST_PARALLEL_LEVEL=2"
                // WORKAROUND_SWDEV_290754: "LLVM_PATH=/opt/rocm/llvm"
                Navi21_build_cmd = "LLVM_PATH=/opt/rocm/llvm CTEST_PARALLEL_LEVEL=2 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check"
            }
            parallel{
                stage('Fp32 Hip All gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_FP32 }
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test, gpu_arch: "gfx908")
                    }
                }
                stage('Fp32 Hip All gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_FP32 }
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test, gpu_arch: "gfx90a:xnack-")
                    }
                }
                // #TODO Code Quality WORKAROUND ROCm 5.1 update
                // stage('Fp32 Hip All gfx90a Xnack+') {
                //     when {
                //         beforeAgent true
                //         expression { params.TARGET_GFX90A && params.DATATYPE_FP32 }
                //     }
                //     agent{ label rocmnode("gfx90a") }
                //     steps{
                //         buildHipClangJobAndReboot(setup_flags: Full_test, gpu_arch: "gfx90a:xnack+", enforce_xnack_on: true)
                //     }
                // }
                stage('Fp16 Hip Install All Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_FP16 }
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Full_test + Fp16_flags, build_env: WORKAROUND_iGemm_936, build_install: "true")
                    }
                }
                stage('Fp32 Hip All Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_FP32 }
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Full_test)
                    }
                }
                stage('Fp32 OpenCL All gfx1030') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_NAVI21 && params.DATATYPE_FP32 }
                    }
                    agent{ label rocmnode("navi21") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', setup_flags: Full_test, build_cmd: Navi21_build_cmd, gpu_arch: "gfx1030")
                    }
                }
                stage('Fp32 Hip All Install gfx1030') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_NAVI21 && params.DATATYPE_FP32 }
                    }
                    agent{ label rocmnode("navi21") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test, build_cmd: Navi21_build_cmd, build_install: "true", gpu_arch: "gfx1030")
                    }
                }
                stage('Fp16 Hip All Install gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_FP16 }
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_env: WORKAROUND_iGemm_936, build_install: "true", gpu_arch: "gfx908")
                    }
                }
                stage('Fp16 Hip All Install gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_FP16 }
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_env: WORKAROUND_iGemm_936, build_install: "true", gpu_arch: "gfx90a:xnack-")
                    }
                }
            }
        }

        stage("Packages") {
            when {
                expression { params.BUILD_PACKAGES && params.TARGET_NOGPU && params.DATATYPE_NA }
            }
            parallel {
                stage('OpenCL Package') {
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', package_build: "true", gpu_arch: "gfx900;gfx906;gfx908;gfx90a", needs_gpu:false)
                    }
                }
                stage("HIP Package /opt/rocm") {
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildHipClangJobAndReboot( package_build: "true", prefixpath: '/opt/rocm', gpu_arch: "gfx900;gfx906;gfx908;gfx90a", needs_gpu:false)
                    }
                }
            }
        }
    }
}

def rocmnode(name) {
    return '(rocmtest || miopen) && (' + name + ')'
}

def miopenCheckout()
{
    checkout([
        $class: 'GitSCM',
        branches: scm.branches,
        doGenerateSubmoduleConfigurations: true,
        extensions: scm.extensions + [[$class: 'SubmoduleOption', parentCredentials: true]],
        userRemoteConfigs: scm.userRemoteConfigs
    ])
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
    def make_targets = conf.get("make_targets","check")
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined -Wno-option-ignored " + conf.get("extradebugflags", "")
    def build_envs = "CTEST_PARALLEL_LEVEL=4 " + conf.get("build_env","")
    def prefixpath = conf.get("prefixpath","/opt/rocm")
    def build_type_debug = (conf.get("build_type",'release') == 'debug')

    def mlir_args = " -DMIOPEN_USE_MLIR=" + conf.get("mlir_build", "ON")
    // WORKAROUND_ISSUE_3192 Disabling MLIR for debug builds since MLIR generates sanitizer errors.
    if (build_type_debug)
    {
        mlir_args = " -DMIOPEN_USE_MLIR=OFF"
    }

    def setup_args = mlir_args + " -DMIOPEN_GPU_SYNC=Off " + conf.get("setup_flags","")
    def build_fin = conf.get("build_fin", "OFF")

    setup_args = setup_args + " -DCMAKE_PREFIX_PATH=${prefixpath} "

    //cmake_env can overwrite default CXX variables.
    def cmake_envs = "CXX=${compiler} CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")

    def package_build = (conf.get("package_build","") == "true")

    if (package_build == true) {
        make_targets = "miopen_gtest package miopen_gtest_check"
        setup_args = " -DMIOPEN_TEST_DISCRETE=OFF " + setup_args
    }

    def miopen_install_path = "${env.WORKSPACE}/install"
    if(conf.get("build_install","") == "true")
    {
        make_targets = 'install ' + make_targets
        setup_args = " -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=${miopen_install_path}" + setup_args
    } else if(package_build == true) {
        setup_args = ' -DBUILD_DEV=Off' + setup_args
    } else {
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

    if(build_type_debug){
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

    if ( build_fin == "ON" )
    {
        setup_args = " -DMIOPEN_INSTALL_CXX_HEADERS=On " + setup_args
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
    def build_cmd = conf.get("build_cmd", "LLVM_PATH=/opt/rocm/llvm ${build_envs} dumb-init make -j\$(nproc) ${make_targets}")
    def execute_cmd = conf.get("execute_cmd", "")

    def cmd = conf.get("cmd", """
            ${pre_setup_cmd}
            ${setup_cmd}
            ${build_cmd}
        """)

    if ( build_fin == "ON" )
    {
        def fin_build_cmd = cmake_fin_build_cmd(miopen_install_path)
        cmd += """
            export RETDIR=\$PWD
            cd ${env.WORKSPACE}/fin
            ${fin_build_cmd}
            cd \$RETDIR
        """
    }

    cmd += """
        ${execute_cmd}
    """

    echo cmd
    sh cmd

    // Only archive from master or develop
    if (package_build == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master" ||
        env.BRANCH_NAME == env.MIOPEN_GOLDEN_PERF_BRANCH || params.PERF_TEST_BRANCH_OVERRIDE)) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
        archiveArtifacts artifacts: "build/*.rpm", allowEmptyArchive: true, fingerprint: true
        stash includes: "build/*tar.gz", name: 'miopen_tar'
    }
}

def cmake_fin_build_cmd(prefixpath){
    def flags = "-DCMAKE_INSTALL_PREFIX=${prefixpath} -DCMAKE_BUILD_TYPE=release"
    def compiler = 'clang++'
    def make_targets = "install"
    def compilerpath = "/opt/rocm/llvm/bin/" + compiler
    def configargs = ""
    if (prefixpath != "")
    {
        configargs = "-DCMAKE_PREFIX_PATH=${prefixpath}"
    }

    def fin_cmd = """
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            rm -rf build
            mkdir build
            cd build
            CXX=${compilerpath} cmake ${configargs} ${flags} ..
            dumb-init make -j\$(nproc) ${make_targets}
    """
    return fin_cmd
}

def getDockerImageName(dockerArgs)
{
    sh "echo ${dockerArgs} > factors.txt"
    def image = "${env.MIOPEN_DOCKER_IMAGE_URL}"
    sh "md5sum Dockerfile requirements.txt dev-requirements.txt >> factors.txt"
    def docker_hash = sh(script: "md5sum factors.txt | awk '{print \$1}' | head -c 6", returnStdout: true)
    sh "rm factors.txt"
    echo "Docker tag hash: ${docker_hash}"
    image = "${image}:ci_${docker_hash}"
    if(params.DOCKER_IMAGE_OVERRIDE != '')
    {
        echo "Overriding the base docker image with ${params.DOCKER_IMAGE_OVERRIDE}"
        image = "${params.DOCKER_IMAGE_OVERRIDE}"
    }
    return image

}

def getDockerImage(Map conf=[:])
{
    env.DOCKER_BUILDKIT=1
    def prefixpath = conf.get("prefixpath", "/opt/rocm") // one image for each prefix 1: /usr/local 2:/opt/rocm
    def gpu_arch = "gfx908;gfx90a;gfx942;gfx1100" // prebuilt dockers should have all the architectures enabled so one image can be used for all stages
    def mlir_build = conf.get("mlir_build", "ON") // always ON
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${prefixpath} --build-arg GPU_ARCHS='\"${gpu_arch}\"' --build-arg USE_MLIR='${mlir_build}' "
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
    echo "Docker Args: ${dockerArgs}"

    def image = getDockerImageName(dockerArgs)

    def dockerImage
    try{
        echo "Pulling down image: ${image}"
        dockerImage = docker.image("${image}")
        dockerImage.pull()
    }
    catch(org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
        echo "The job was cancelled or aborted"
        throw e
    }
    catch(Exception ex)
    {
        dockerImage = docker.build("${image}", "${dockerArgs} .")
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            dockerImage.push()
        }
    }
    return [dockerImage, image]
}

def buildHipClangJob(Map conf=[:]){
        show_node_info()
        miopenCheckout()
        env.HSA_ENABLE_SDMA=0
        env.DOCKER_BUILDKIT=1
        def image
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1"
        }

        def variant = env.STAGE_NAME

        def needs_gpu = conf.get("needs_gpu", true)
        def lfs_pull = conf.get("lfs_pull", false)

        def retimage
        gitStatusWrapper(credentialsId: "${env.miopen_git_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCm', repo: 'MIOpen') {
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

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 420, unit:'MINUTES')
                {
                    if (lfs_pull) {
                        sh "git lfs pull --exclude="
                    }

                    cmake_build(conf)
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
        cleanWs()
    }
    catch(e){
        echo "throwing error exception for the stage"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (conf.get("needs_reboot", true)) {
            reboot()
        }
    }
}

def RunPerfTest(Map conf=[:]){
    def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    try {
        (retimage, image) = getDockerImage(conf)
        withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
        timeout(time: 600, unit: 'MINUTES')
        {
            unstash 'miopen_tar'
            sh "tar -zxvf build/miopen-hip-*-Linux-runtime.tar.gz"
            ld_lib="${env.WORKSPACE}/opt/rocm/lib"
            def filename = conf.get("filename", "")
            if (env.BRANCH_NAME == env.MIOPEN_GOLDEN_PERF_BRANCH || params.PERF_TEST_BRANCH_OVERRIDE){
                if(params.PERF_TEST_OVERRIDE != '')
                {
                    echo "Appending MIOpenDriver cmd env vars: ${params.PERF_TEST_OVERRIDE}"
                    sh "export LD_LIBRARY_PATH=${ld_lib} && ${env.WORKSPACE}/opt/rocm/bin/test_perf.py  --filename ${filename} --install_path ${env.WORKSPACE}/opt/rocm --override ${params.PERF_TEST_OVERRRIDE}"
                }else
                {
                    sh "export LD_LIBRARY_PATH=${ld_lib} && ${env.WORKSPACE}/opt/rocm/bin/test_perf.py  --filename ${filename} --install_path ${env.WORKSPACE}/opt/rocm"
                }
                sh "export LD_LIBRARY_PATH=${ld_lib} && ${env.WORKSPACE}/opt/rocm/bin/test_perf.py  --filename ${filename} --install_path ${env.WORKSPACE}/opt/rocm"
                jenkins_url = "${env.artifact_path}/${env.MIOPEN_GOLDEN_PERF_BRANCH}/lastSuccessfulBuild/artifact"
                try {
                    sh "rm -rf ${env.WORKSPACE}/opt/rocm/bin/old_results/"
                    sh "wget -P ${env.WORKSPACE}/opt/rocm/bin/old_results/ ${jenkins_url}/opt/rocm/bin/perf_results/${filename}"
                }
                catch (Exception err){
                    currentBuild.result = 'SUCCESS'
                }
            }

            archiveArtifacts artifacts: "opt/rocm/bin/perf_results/${filename}", allowEmptyArchive: true, fingerprint: true
            try{
              if (env.BRANCH_NAME != env.MIOPEN_GOLDEN_PERF_BRANCH){
                  sh "${env.WORKSPACE}/opt/rocm/bin/test_perf.py --compare_results --old_results_path ${env.WORKSPACE}/opt/rocm/bin/old_results --filename ${filename}"
              }
            }
            catch (Exception err){
                currentBuild.result = 'SUCCESS'
            }
            cleanWs()
        }
        }
    }
    catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
        echo "The job was cancelled or aborted"
        throw e
    }
}


def CheckPerfDbValid(Map conf=[:]){
    def pdb_image = buildHipClangJob(conf)
    pdb_image.inside(){
        dir(path: "$WORKSPACE"){
            sh "ls install/bin/"
            sh "MIOPEN_LOG_LEVEL=4 LD_LIBRARY_PATH='install/lib:/opt/rocm/lib/' install/bin/fin -i fin/tests/pdb_check_all.json -o pdb_valid_err.json"
            archiveArtifacts "pdb_valid_err.json"
            sh "grep clear pdb_valid_err.json"
            def has_error = sh (
                script: "echo \$?",
                returnStdout: true
            ).trim()
            assert has_error.toInteger() == 0
        }
    }
}

/// Stage name format:
/// [DataType] Backend[/Compiler] BuildType [TestSet] [Target]
///
/// The only mandatory elements are Backend and BuildType; others are optional.
///
/// DataType := { Fp16 | Bf16 | Int8 | Fp32 }
/// Backend := { Hip | HipNoGPU}
/// Compiler := { Clang* | GCC* }
///   * "Clang" is the default for the Hip backend, and implies hip-clang compiler.
///   * The default compiler is usually not specified.
/// BuildType := { Release* | Debug | Install } [ BuildTypeModifier ]
///   * BuildTypeModifier := { NOCOMGR | Embedded | Static | Normal-Find | Fast-Find
///                            NOCK | NOMLIR | Tensile | Tensile-Latest | Package | ... }
/// TestSet := { All | Smoke* | <Performance Dataset> | Build-only }
///   * "All" corresponds to "cmake -DMIOPEN_TEST_ALL=On".
///   * "Smoke" (-DMIOPEN_TEST_ALL=Off) is the default and usually not specified.
///   * "Performance Dataset" is a performance test with a specified dataset.
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
            name: "BUILD_FULL_TESTS",
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
            defaultValue: false,
            description: "")
        booleanParam(
            name: "TARGET_VEGA20",
            defaultValue: false,
            description: "")
        booleanParam(
            name: "TARGET_GFX908",
            defaultValue: env.BRANCH_NAME == "develop" ? true : false,
            description: "")
        booleanParam(
            name: "TARGET_GFX90A",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "TARGET_GFX94X",
            defaultValue: false,
            description: "")
        booleanParam(
            name: "TARGET_NAVI21",
            defaultValue: false,
            description: "")
        booleanParam(
            name: "TARGET_NAVI32",
            defaultValue: false,
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
        booleanParam(
            name: "PERF_TEST",
            defaultValue: false,
            description: "Enable performance testing stages")
        booleanParam(
            name: "PERF_TEST_FP16",
            defaultValue: false,
            description: "Enable performance testing stages")
        booleanParam(
            name: "PERF_TEST_FP32",
            defaultValue: false,
            description: "Enable performance testing stages")
        booleanParam(
            name: "PERF_TEST_BRANCH_OVERRIDE",
            defaultValue: false,
            description: "Enable performance testing stages")
        booleanParam(
            name: "DBSYNC_TEST",
            defaultValue: true,
            description: "Enable database synchronization testing stages")
        string(name: "PERF_TEST_OVERRIDE",
            defaultValue: '',
            description: "Add extra env vars for the MIOpenDriver cmd, comma separated")
        string(name: "DOCKER_IMAGE_OVERRIDE",
            defaultValue: '',
            description: "")
        booleanParam(
            name: "WORKAROUND__TARGET_GFX94X_MINIMUM_TEST_ENABLE",
            defaultValue: false,
            description: "")
    }

    environment{
        extra_log_env   = " MIOPEN_LOG_LEVEL=5 "
        Fp16_flags      = " -DMIOPEN_TEST_HALF=On"
        Bf16_flags      = " -DMIOPEN_TEST_BFLOAT16=On"
        Int8_flags      = " -DMIOPEN_TEST_INT8=On"
        Full_test       = " -DMIOPEN_TEST_ALL=On"
        Smoke_targets   = " check MIOpenDriver"
        NOCOMGR_flags   = " -DMIOPEN_USE_COMGR=Off"
        NOMLIR_flags    = " -DMIOPEN_USE_MLIR=Off"
    }
    triggers{

        cron(env.BRANCH_NAME == env.NIGHTLY_BRANCH ? env.NIGHTLY_SCHEDULE : '')
    }
    stages{
        stage('Build Docker'){
            when {
                expression { params.BUILD_DOCKER && params.TARGET_NOGPU && params.DATATYPE_NA }
            }
            agent{ label rocmnode("gfx90a") }
            steps{
                getDockerImage()
            }
        }
        stage("Packages") {
            when {
                expression { params.BUILD_PACKAGES && params.TARGET_NOGPU && params.DATATYPE_NA }
            }
            parallel {
                stage("HIP Package") {
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildHipClangJobAndReboot( package_build: "true", needs_gpu:false, needs_reboot:false)
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
                        setup_cmd = "CXX='/opt/rocm/llvm/bin/clang++' cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On .. "
                        build_cmd = "make -j\$(nproc) -k analyze"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: setup_cmd, build_cmd: build_cmd, needs_gpu:false, needs_reboot:false)
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
                                | grep -v -E '(build/)|(install/)|(fin/)' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-12 -style=file {} | diff - {}\'"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd, needs_gpu:false, needs_reboot:false)
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
                        buildHipClangJob( build_type: 'debug', setup_flags: HipNoGPU_flags, build_cmd: build_cmd, needs_gpu:false, needs_reboot:false)
                    }
                }
                stage('Tuna Fin Build Test') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                      fin_flags = "-DMIOPEN_BACKEND=HIPNOGPU"
                    }
                    steps{
		      buildHipClangJobAndReboot(setup_flags: fin_flags, make_targets: "all", build_fin: "ON", needs_gpu:false, needs_reboot:false, build_install: "true")
                    }
                }
            }
        }
        stage("Smoke Fp32") {
            when {
                expression { params.BUILD_SMOKE_FP32 && params.DATATYPE_FP32 }
            }
            parallel{
                stage('Fp32 Hip gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(make_targets: Smoke_targets, build_install: "true")
                    }
                }
                stage('Fp32 Hip Debug gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(build_type: 'debug', make_targets: Smoke_targets, build_install: "true")
                    }
                }
                stage('Fp32 Hip Debug gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(build_type: 'debug', make_targets: Smoke_targets, build_install: "true")
                    }
                }
                stage('Fp32 Hip Debug gfx94X') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX94X || params.WORKAROUND__TARGET_GFX94X_MINIMUM_TEST_ENABLE }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx94X") }
                    steps{
                        buildHipClangJobAndReboot(build_type: 'debug', make_targets: Smoke_targets, needs_reboot:false, build_install: "true")
                    }
                }
            }
        }
        stage("Smoke Aux 1") {
            when {
                expression { params.BUILD_SMOKE_AUX1 && params.DATATYPE_FP32 }
            }
            parallel{
                stage('Fp32 Hip Debug NOCOMGR gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        // Can be removed altogether with when WORKAROUND_SWDEV_290754.
                        NOCOMGR_build_cmd = "CTEST_PARALLEL_LEVEL=4 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( build_type: 'debug', setup_flags: NOCOMGR_flags, build_cmd: NOCOMGR_build_cmd, test_flags: ' --verbose ', build_install: "true")
                    }
                }
                stage('Fp32 Hip Debug NOMLIR gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        // Can be removed altogether with when WORKAROUND_SWDEV_290754.
                        NOMLIR_build_cmd = "CTEST_PARALLEL_LEVEL=4 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( build_type: 'debug', setup_flags: NOMLIR_flags, build_cmd: NOMLIR_build_cmd, test_flags: ' --verbose ', build_install: "true")
                    }
                }
                stage('Fp32 Hip Debug NOCK gfx90a Build-Only') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot( build_type: 'debug', setup_flags: "-DMIOPEN_USE_COMPOSABLEKERNEL=Off", make_targets: "", build_install: "true")
                    }
                }
                stage('Fp32 Hip Debug Embedded Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("vega20") }
                    environment{
                        Embedded_flags = "-DMIOPEN_EMBED_DB='gfx906_60'"
                    }
                    steps{
                        buildHipClangJobAndReboot( build_type: 'debug', setup_flags: Embedded_flags, build_env: extra_log_env, test_flags: ' --verbose ', build_install: "true")
                    }
                }
                stage('Fp32 Hip Static gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: "-DBUILD_SHARED_LIBS=Off", mlir_build: 'OFF', build_install: "true")
                    }
                }
                stage('Fp32 Hip Normal-Find gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        make_targets = "test_conv2d"
                        execute_cmd = "bin/test_conv2d --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot(make_targets: make_targets, execute_cmd: execute_cmd, find_mode: "Normal", build_install: "true")
                    }
                }
                stage('Fp32 Hip Fast-Find gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        make_targets =   "test_conv2d"
                        execute_cmd = "MIOPEN_FIND_MODE=2 CTEST_PARALLEL_LEVEL=4 bin/test_conv2d --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot( make_targets: make_targets, execute_cmd: execute_cmd, build_install: "true")
                    }
                }
                stage('Fp32 Hip gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot()
                    }
                }
                stage('Fp32 Hip SqlitePerfdb gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(make_targets: Smoke_targets, setup_flags: "-DMIOPEN_USE_SQLITE_PERF_DB=On", build_install: "true")
                    }
                }
                stage('Fp32 Hip Fin Interface gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: "-DMIOPEN_ENABLE_FIN_INTERFACE=On",
                                                  make_targets: "test_unit_FinInterface",
                                                  execute_cmd: "bin/test_unit_FinInterface")
                    }
                }
            }
        }
        stage("Smoke Fp16/Bf16/Int8") {
            when {
                expression { params.BUILD_SMOKE_FP16_BF16_INT8 }
            }
            parallel{
                stage('Fp16 Hip Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Fp16_flags, make_targets: Smoke_targets, build_install: "true")
                    }
                }
                stage('Bf16 Hip Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_BF16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags, make_targets: Smoke_targets, build_install: "true")
                    }
                }
                stage('Fp16 Hip gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Fp16_flags, make_targets: Smoke_targets, build_install: "true")
                    }
                }
                stage('Bf16 Hip gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_BF16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags, make_targets: Smoke_targets, build_install: "true")
                    }
                }
                stage('Fp16 Hip gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Fp16_flags, make_targets: Smoke_targets, build_install: "true")
                    }
                }
                stage('Bf16 Hip gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_BF16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags, make_targets: Smoke_targets, build_install: "true")
                    }
                }
                stage('Fp16 Hip gfx94X') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX94X && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx94X") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Fp16_flags, make_targets: Smoke_targets, needs_reboot:false, build_install: "true")
                    }
                }
                stage('Bf16 Hip gfx94X') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX94X && params.DATATYPE_BF16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx94X") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags, make_targets: Smoke_targets, needs_reboot:false, build_install: "true")
                    }
                }
            }
        }
        stage("Full Tests") {
            when {
                expression { params.BUILD_FULL_TESTS}
            }
            environment{
                // WORKAROUND_ISSUE_1148: "CTEST_PARALLEL_LEVEL=2"
                // WORKAROUND_SWDEV_290754: "LLVM_PATH=/opt/rocm/llvm"
                Navi21_build_cmd = "LLVM_PATH=/opt/rocm/llvm CTEST_PARALLEL_LEVEL=2 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check"
            }
            parallel{
                stage('Dbsync gfx908') {
                    when {
                        beforeAgent true
                        expression { params.DBSYNC_TEST && params.TARGET_GFX908 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(lfs_pull: true,
                                                  setup_flags: "-DMIOPEN_TEST_DBSYNC=1",
                                                  make_targets: 'test_db_sync',
                                                  execute_cmd: './bin/test_db_sync',
                                                  needs_gpu:false,
                                                  needs_reboot:false,
                                                  build_install: "true")
                    }
                }
                stage('Dbsync gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.DBSYNC_TEST && params.TARGET_GFX90A }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(lfs_pull: true,
                                                  setup_flags: "-DMIOPEN_TEST_DBSYNC=1",
                                                  make_targets: 'test_db_sync',
                                                  execute_cmd: './bin/test_db_sync',
                                                  needs_gpu:false,
                                                  needs_reboot:false,
                                                  build_install: "true")
                    }
                }
                stage('Dbsync gfx942') {
                    when {
                        beforeAgent true
                        expression { params.DBSYNC_TEST && (params.TARGET_GFX94X || params.WORKAROUND__TARGET_GFX94X_MINIMUM_TEST_ENABLE) }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        buildHipClangJobAndReboot(lfs_pull: true,
                                                  setup_flags: "-DMIOPEN_TEST_DBSYNC=1",
                                                  make_targets: 'test_db_sync',
                                                  execute_cmd: './bin/test_db_sync',
                                                  needs_gpu:false,
                                                  needs_reboot:false,
                                                  build_install: "true")
                    }
                }
                stage('Int8 HIP All Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_INT8 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Int8_flags + Full_test)
                    }
                }
                stage('Bf16 Hip Install All gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_BF16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: "true")
                    }
                }
                stage('Bf16 Hip Install All gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_BF16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: "true")
                    }
                }
                stage('Bf16 Hip Install All gfx94X') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX94X && params.DATATYPE_BF16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx94X") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: "true", needs_reboot:false)
                    }
                }
                stage('Fp16 Hip All gfx1030') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_NAVI21 && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("navi21") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_cmd: Navi21_build_cmd)
                    }
                }
                stage('Fp16 Hip All gfx1101') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_NAVI32 && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("navi32") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags)
                    }
                }
                stage('Fp32 Hip All gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_FP32 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test)
                    }
                }
                stage('Fp32 Hip All gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_FP32 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test)
                    }
                }
                // stage('Fp32 Hip All gfx90a Xnack+') {
                //     when {
                //         beforeAgent true
                //         expression { params.TARGET_GFX90A && params.DATATYPE_FP32 }
                //     }
                //     agent{ label rocmnode("gfx90a") }
                //     steps{
                //         buildHipClangJobAndReboot(setup_flags: Full_test, enforce_xnack_on: true)
                //     }
                // }
                stage('Fp32 Hip All gfx94X') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX94X && params.DATATYPE_FP32 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx94X") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test, needs_reboot:false)
                    }
                }
                stage('Fp16 Hip Install All Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Full_test + Fp16_flags, build_install: "true")
                    }
                }
                stage('Fp32 Hip All Vega20') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_VEGA20 && params.DATATYPE_FP32 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Full_test)
                    }
                }
                stage('Fp32 Hip All Install gfx1030') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_NAVI21 && params.DATATYPE_FP32 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("navi21") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test, build_cmd: Navi21_build_cmd, build_install: "true")
                    }
                }
                stage('Fp32 Hip All Install gfx1101') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_NAVI32 && params.DATATYPE_FP32 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("navi32") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test, build_install: "true")
                    }
                }
                stage('Fp16 Hip All Install gfx908') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX908 && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_install: "true")
                    }
                }
                stage('Fp16 Hip All Install gfx90a') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX90A && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_install: "true")
                    }
                }
                stage('Fp16 Hip All Install gfx94X') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX94X && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx94X") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_install: "true", needs_reboot:false)
                    }
                }
            }
        }
        stage("Performance Tests - gfx90a") {
            when {
                expression {params.PERF_TEST && params.TARGET_GFX90A}
            }
            parallel{
                stage('Fp32 BS128 Hip Performance Resnet50_v1.5 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1.5_FP32_BS128.txt" )
                    }
                }
                stage('Fp32 BS256 Hip Performance Resnet50_v1.5 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1.5_FP32_BS256.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance Resnet50_v1.5 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1.5_FP32_BS512.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance Resnet50_v1.5 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1.5_FP16_BS128.txt" )
                    }
                }
                stage('Fp16 BS256 Hip Performance Resnet50_v1.5 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1.5_FP16_BS256.txt" )
                    }
                }
                stage('Fp16 BS512 Hip Performance Resnet50_v1.5 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1.5_FP16_BS512.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance Alexnet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Alexnet_v1_FP16_BS128.txt" )
                    }
                }
                stage('Fp16 BS512 Hip Performance Alexnet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Alexnet_v1_FP16_BS512.txt" )
                    }
                }
                stage('Fp32 BS4 Hip Performance Alexnet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Alexnet_v1_FP32_BS4.txt" )
                    }
                }
                stage('Fp32 BS64 Hip Performance Alexnet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Alexnet_v1_FP32_BS64.txt" )
                    }
                }
                stage('Fp32 BS128 Hip Performance Alexnet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Alexnet_v1_FP32_BS128.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance Alexnet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Alexnet_v1_FP32_BS512.txt" )
                    }
                }
                stage('Fp16 BS256 Hip Performance Densenet201_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Densenet201_v1_FP16_BS256.txt" )
                    }
                }
                stage('Fp32 BS256 Hip Performance Densenet201_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Densenet201_v1_FP32_BS256.txt" )
                    }
                }
                stage('Fp16 BS256 Hip Performance Densenet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Densenet_v1_FP16_BS256.txt" )
                    }
                }
                stage('Fp32 BS256 Hip Performance Densenet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Densenet_v1_FP32_BS256.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance Googlenet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Googlenet_v1_FP16_BS128.txt" )
                    }
                }
                stage('Fp16 BS512 Hip Performance Googlenet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Googlenet_v1_FP16_BS512.txt" )
                    }
                }
                stage('Fp32 BS128 Hip Performance Googlenet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Googlenet_v1_FP32_BS128.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance Googlenet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Googlenet_v1_FP32_BS512.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance Inception3_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Inception3_v1_FP16_BS128.txt" )
                    }
                }
                stage('Fp32 BS128 Hip Performance Inception3_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Inception3_v1_FP32_BS128.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance Inception3_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Inception3_v1_FP32_BS512.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance Inception4_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Inception4_v1_FP16_BS128.txt" )
                    }
                }
                stage('Fp16 BS512 Hip Performance Inception4_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Inception4_v1_FP16_BS512.txt" )
                    }
                }
                stage('Fp32 BS128 Hip Performance Inception4_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Inception4_v1_FP32_BS128.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance Inception4_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Inception4_v1_FP32_BS512.txt" )
                    }
                }
                stage('Fp32 BS4 Hip Performance Mobilenet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Mobilenet_v1_FP32_BS4.txt" )
                    }
                }
                stage('Fp32 BS64 Hip Performance Mobilenet_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Mobilenet_v1_FP32_BS64.txt" )
                    }
                }
                stage('Fp16 BS32 Hip Performance Resnet101_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet101_v1_FP16_BS32.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance Resnet101_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet101_v1_FP16_BS128.txt" )
                    }
                }
                stage('Fp16 BS256 Hip Performance Resnet101_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet101_v1_FP16_BS256.txt" )
                    }
                }
                stage('Fp16 BS512 Hip Performance Resnet101_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet101_v1_FP16_BS512.txt" )
                    }
                }
                stage('Fp32 BS128 Hip Performance Resnet101_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet101_v1_FP32_BS128.txt" )
                    }
                }
                stage('Fp32 BS256 Hip Performance Resnet101_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet101_v1_FP32_BS256.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance Resnet101_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet101_v1_FP32_BS512.txt" )
                    }
                }
                stage('Fp16 BS256 Hip Performance Resnet152_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet152_v1_FP16_BS256.txt" )
                    }
                }
                stage('Fp32 BS256 Hip Performance Resnet152_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet152_v1_FP32_BS256.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance Resnet152_v2 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet152_v2_FP16_BS128.txt" )
                    }
                }
                stage('Fp16 BS512 Hip Performance Resnet152_v2 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet152_v2_FP16_BS512.txt" )
                    }
                }
                stage('Fp32 BS128 Hip Performance Resnet152_v2 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet152_v2_FP32_BS128.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance Resnet152_v2 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet152_v2_FP32_BS512.txt" )
                    }
                }
                stage('Fp16 BS32 Hip Performance Resnet50_v1 gfx90a'){

                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1_FP16_BS32.txt" )
                    }
                }
                stage('Fp16 BS64 Hip Performance Resnet50_v1 gfx90a'){

                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1_FP16_BS64.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance Resnet50_v1 gfx90a'){

                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1_FP16_BS128.txt" )
                    }
                }
                stage('Fp16 BS256 Hip Performance Resnet50_v1 gfx90a'){

                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1_FP16_BS256.txt" )
                    }
                }
                stage('Fp16 B512 Hip Performance Resnet50_v1 gfx90a'){

                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1_FP16_BS512.txt" )
                    }
                }
                stage('Fp32 BS128 Hip Performance Resnet50_v1 gfx90a'){

                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1_FP32_BS128.txt" )
                    }
                }
                stage('Fp32 BS256 Hip Performance Resnet50_v1 gfx90a'){

                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1_FP32_BS256.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance Resnet50_v1 gfx90a'){

                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Resnet50_v1_FP32_BS512.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance Shufflenet_v2 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "Shufflenet_v2_FP16_BS128.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance SSD_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "SSD_v1_FP16_BS128.txt" )
                    }
                }
                stage('Fp32 BS128 Hip Performance SSD_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "SSD_v1_FP32_BS128.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance VGG11_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG11_v1_FP16_BS128.txt" )
                    }
                }
                stage('Fp16 BS256 Hip Performance VGG11_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG11_v1_FP16_BS256.txt" )
                    }
                }
                stage('Fp16 BS512 Hip Performance VGG11_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG11_v1_FP16_BS512.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance VGG11_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG11_v1_FP32_BS512.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance VGG16_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG16_v1_FP16_BS128.txt" )
                    }
                }
                stage('Fp32 BS4 Hip Performance VGG16_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG16_v1_FP32_BS4.txt" )
                    }
                }
                stage('Fp32 BS64 Hip Performance VGG16_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG16_v1_FP32_BS64.txt" )
                    }
                }
                stage('Fp32 BS128 Hip Performance VGG16_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG16_v1_FP32_BS128.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance VGG16_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG16_v1_FP32_BS512.txt" )
                    }
                }
                stage('Fp16 BS128 Hip Performance VGG19_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG19_v1_FP16_BS128.txt" )
                    }
                }
                stage('Fp16 BS512 Hip Performance VGG19_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP16}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG19_v1_FP16_BS512.txt" )
                    }
                }
                stage('Fp32 BS128 Hip Performance VGG19_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG19_v1_FP32_BS128.txt" )
                    }
                }
                stage('Fp32 BS512 Hip Performance VGG19_v1 gfx90a'){
                    when {
                        expression {params.PERF_TEST_FP32}
                    }
                    agent{ label rocmnode("austin")}
                    steps{
                        RunPerfTest(gpu_arch: "gfx90a", filename: "VGG19_v1_FP32_BS512.txt" )
                    }
                }
            }
        }
    }
}

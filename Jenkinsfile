def repoName = scm.getUserRemoteConfigs()[0].getUrl().tokenize('/').last().replace('.git', '')

def repoDir = "./"
if (repoName == "rocm-libraries") {
    repoDir = "/projects/miopen"
}

def rocmnode(name) {
    return '(rocmtest || miopen) && (' + name + ')'
}

def get_branch_name(){
    def shared_library_branch = scm.branches[0].name
    if (shared_library_branch .contains("*/")) {
        shared_library_branch  = shared_library_branch.split("\\*/")[1]
    }
    echo "${shared_library_branch}"
    return shared_library_branch
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
/// Target := { gfx908 | gfx90a | gfx942 } [ Xnack+ ]

def utils

def withWorkingDir(Closure body) {
    checkout scm
    dir("${env.WORKSPACE}/${env.REPO_DIR}") {
        body()
    }
}

def runDbSyncJob(def utils)
{
    script {
        withWorkingDir {
            utils.buildHipClangJobAndReboot(dvc_pull: true,
                                setup_flags: "-DMIOPEN_TEST_DBSYNC=1",
                                make_targets: 'test_db_sync',
                                execute_cmd: './bin/test_db_sync',
                                needs_gpu:false,
                                needs_reboot:false,
                                build_install: true)
        }
    }
}

//launch develop branch nightly jobs
CRON_SETTINGS = BRANCH_NAME == "develop" ? '''0 0 * * * % RUN_NIGHTLY_TESTS=true;BUILD_PACKAGE_AND_CHECKS=false;BUILD_FULL_TESTS=false;TARGET_GFX908=true;TARGET_GFX90A=true;TARGET_GFX942=true''' : ""

pipeline {
    agent none
    options {
        skipDefaultCheckout()
        parallelsAlwaysFailFast()
        // disable stage-wise timeout due to long wait with queue (limited resources)
        // timeout(time: 90, unit:'MINUTES')
    }
    triggers{
        parameterizedCron(CRON_SETTINGS)
    }
    parameters {
        booleanParam(
            name: "BUILD_DOCKER",
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
            name: "BUILD_PACKAGE_AND_CHECKS",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "TARGET_NOGPU",
            defaultValue: true,
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
            name: "TARGET_GFX942",
            defaultValue: env.BRANCH_NAME == "develop" ? true : false,
            description: "")
        booleanParam(
            name: "TARGET_NAVI32",
            defaultValue: false,
            description: "Navi3 currently fails to build with instruction not supported on this GPU error")
        booleanParam(
            name: "TARGET_NAVI4",
            defaultValue: false,
            description: "Navi4 currently fails to build with instruction not supported on this GPU error")
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
            name: "DBSYNC_TEST",
            defaultValue: true,
            description: "Enable database synchronization testing stages")
        string(name: "DOCKER_IMAGE_OVERRIDE",
            defaultValue: '',
            description: "")
        booleanParam(
            name: "WORKAROUND__TARGET_GFX942_MINIMUM_TEST_ENABLE",
            defaultValue: false,
            description: "")
        booleanParam(
            name: "USE_SCCACHE_DOCKER",
            defaultValue: true,
            description: "Use the sccache for building CK in the Docker Image (default: ON)")
        booleanParam(
            name: "RUN_NIGHTLY_TESTS",
            defaultValue: false,
            description: "Run the nightly tests (default: OFF)")
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
        REPO_DIR        = "${repoDir}"
        REPO_NAME       = "${repoName}"
    }
    stages{
        stage('Build Docker'){
            when {
                expression { params.BUILD_DOCKER && params.TARGET_NOGPU && params.DATATYPE_NA }
            }
            agent{ label rocmnode("gfx90a") }
            steps{
                script {
                    withWorkingDir {
                        utils = load "vars/utils.groovy"
                        utils.getDockerImage()
                    }
                }
            }
        }
        stage("Package and Static checks") {
            when {
                expression { params.BUILD_PACKAGE_AND_CHECKS && params.TARGET_NOGPU && params.DATATYPE_NA }
            }
            parallel
            {
                stage("HIP Package") {
                    agent{ label rocmnode("nogpu") }
                    steps {
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot( package_build:true, needs_gpu:false, needs_reboot:false)
                            }
                        }
                    }
                }
                stage('Clang Format') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        execute_cmd = "find ${repoDir} -iname \'*.h\' \
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd, needs_gpu:false, needs_reboot:false)
                            }
                        }
                    }
                }
                stage('Check GTest Format') {
                    agent { label rocmnode("nogpu") }
                    when {
                        changeset "**/test/gtest/**"
                    }
                    steps {
                        script {
                            withWorkingDir {
                                sh 'cd ./test/utils && python3 gtest_formating_checks.py'
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJob( build_type: 'debug', setup_flags: HipNoGPU_flags, build_cmd: build_cmd, needs_gpu:false, needs_reboot:false)
                            }
                        }
                    }
                }
                stage('Tuna Fin Build Test')
                {
                    agent{ label rocmnode("nogpu") }
                    environment{
                      fin_flags = "-DMIOPEN_BACKEND=HIPNOGPU"
                    }
                    steps{
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: fin_flags, make_targets: "all", build_fin: "ON", needs_gpu:false, needs_reboot:false, build_install: true)
                            }
                        }
                    }
                }
            }
        }
        stage("Full Tests") {
            when {
                expression { params.BUILD_FULL_TESTS }
            }
            parallel{
                stage('Hip Tidy') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        setup_cmd = "CXX='/opt/rocm/llvm/bin/clang++' cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On .. "
                        build_cmd = "make -j\$(nproc) -k analyze"
                    }
                    steps{
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_cmd: setup_cmd, build_cmd: build_cmd, needs_gpu:false, needs_reboot:false)
                            }
                        }
                    }
                }
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
                        runDbSyncJob(utils)
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
                        runDbSyncJob(utils)
                    }
                }
                stage('Dbsync gfx942') {
                    when {
                        beforeAgent true
                        expression { params.DBSYNC_TEST && (params.TARGET_GFX942 || params.WORKAROUND__TARGET_GFX942_MINIMUM_TEST_ENABLE) }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        runDbSyncJob(utils)
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: true)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: true)
                            }
                        }
                    }
                }
                stage('Bf16 Hip Install All gfx942') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX942 && params.DATATYPE_BF16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: true, needs_reboot:false)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_install: true)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_install: true)
                            }
                        }
                    }
                }
                stage('Fp16 Hip All Install gfx942') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX942 && params.DATATYPE_FP16 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_install: true, needs_reboot:false)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Full_test)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Full_test)
                            }
                        }
                    }
                }
                stage('Fp32 Hip All gfx942') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX942 && params.DATATYPE_FP32 }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Full_test, needs_reboot:false)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: Full_test, build_install: true)
                            }
                        }
                    }
                }
            }
        }
        stage("Nightly Tests") {
            when {
                expression { params.RUN_NIGHTLY_TESTS }
            }
            parallel{
                stage('Mark Build As Nightly') {
                    agent{ label rocmnode("nogpu") }
                    steps{
                        script {
                            withWorkingDir {
                                // Adds a comment under the jenkins build number so you can tell it is a nightly build.
                                currentBuild.description = "Nightly Build"
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot( build_type: 'debug', setup_flags: NOMLIR_flags, build_cmd: NOMLIR_build_cmd, test_flags: ' --verbose ', build_install: true)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot( build_type: 'debug', setup_flags: "-DMIOPEN_USE_COMPOSABLEKERNEL=Off", make_targets: "", build_install: true)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot( setup_flags: "-DBUILD_SHARED_LIBS=Off", mlir_build: 'OFF', build_install: true)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(make_targets: make_targets, execute_cmd: execute_cmd, find_mode: "Normal", build_install: true)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot( make_targets: make_targets, execute_cmd: execute_cmd, build_install: true)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(make_targets: Smoke_targets, setup_flags: "-DMIOPEN_USE_SQLITE_PERF_DB=On", build_install: true)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(setup_flags: "-DMIOPEN_ENABLE_FIN_INTERFACE=On",
                                                            make_targets: "test_unit_FinInterface",
                                                            execute_cmd: "bin/test_unit_FinInterface")
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(build_type: 'debug', make_targets: Smoke_targets, build_install: true)
                            }
                        }
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
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(build_type: 'debug', make_targets: Smoke_targets, build_install: true)
                            }
                        }
                    }
                }
                stage('Fp32 Hip Debug gfx942') {
                    when {
                        beforeAgent true
                        expression { params.TARGET_GFX942 || params.WORKAROUND__TARGET_GFX942_MINIMUM_TEST_ENABLE }
                    }
                    options {
                        retry(2)
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        script {
                            withWorkingDir {
                                utils.buildHipClangJobAndReboot(build_type: 'debug', make_targets: Smoke_targets, needs_reboot:false, build_install: true)
                            }
                        }
                    }
                }
            }
        }
    }
}

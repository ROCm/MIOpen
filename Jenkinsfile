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

library "jenkins-shared@${get_branch_name()}"

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
            name: "DBSYNC_TEST",
            defaultValue: true,
            description: "Enable database synchronization testing stages")
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
                script {
                utils.getDockerImage()
                }
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
                        script {
                            utils.buildHipClangJobAndReboot( package_build:true, needs_gpu:false, needs_reboot:false)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_cmd: setup_cmd, build_cmd: build_cmd, needs_gpu:false, needs_reboot:false)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd, needs_gpu:false, needs_reboot:false)
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
                        utils.buildHipClangJob( build_type: 'debug', setup_flags: HipNoGPU_flags, build_cmd: build_cmd, needs_gpu:false, needs_reboot:false)
                        }
                    }
                }
                stage('Tuna Fin Build Test') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                      fin_flags = "-DMIOPEN_BACKEND=HIPNOGPU"
                    }
                    steps{
                        script {
		                    utils.buildHipClangJobAndReboot(setup_flags: fin_flags, make_targets: "all", build_fin: "ON", needs_gpu:false, needs_reboot:false, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(make_targets: Smoke_targets, build_install: true)
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
                            utils.buildHipClangJobAndReboot(build_type: 'debug', make_targets: Smoke_targets, build_install: true)
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
                            utils.buildHipClangJobAndReboot(build_type: 'debug', make_targets: Smoke_targets, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(build_type: 'debug', make_targets: Smoke_targets, needs_reboot:false, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot( build_type: 'debug', setup_flags: NOCOMGR_flags, build_cmd: NOCOMGR_build_cmd, test_flags: ' --verbose ', build_install: true)
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
                            utils.buildHipClangJobAndReboot( build_type: 'debug', setup_flags: NOMLIR_flags, build_cmd: NOMLIR_build_cmd, test_flags: ' --verbose ', build_install: true)
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
                            utils.buildHipClangJobAndReboot( build_type: 'debug', setup_flags: "-DMIOPEN_USE_COMPOSABLEKERNEL=Off", make_targets: "", build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot( build_type: 'debug', setup_flags: Embedded_flags, build_env: extra_log_env, test_flags: ' --verbose ', build_install: true)
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
                            utils.buildHipClangJobAndReboot( setup_flags: "-DBUILD_SHARED_LIBS=Off", mlir_build: 'OFF', build_install: true)
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
                            utils.buildHipClangJobAndReboot(make_targets: make_targets, execute_cmd: execute_cmd, find_mode: "Normal", build_install: true)
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
                            utils.buildHipClangJobAndReboot( make_targets: make_targets, execute_cmd: execute_cmd, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot()
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
                            utils.buildHipClangJobAndReboot(make_targets: Smoke_targets, setup_flags: "-DMIOPEN_USE_SQLITE_PERF_DB=On", build_install: true)
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
                            utils.buildHipClangJobAndReboot(setup_flags: "-DMIOPEN_ENABLE_FIN_INTERFACE=On",
                                                            make_targets: "test_unit_FinInterface",
                                                            execute_cmd: "bin/test_unit_FinInterface")
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot( setup_flags: Fp16_flags, make_targets: Smoke_targets, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_flags: Bf16_flags, make_targets: Smoke_targets, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot( setup_flags: Fp16_flags, make_targets: Smoke_targets, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_flags: Bf16_flags, make_targets: Smoke_targets, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot( setup_flags: Fp16_flags, make_targets: Smoke_targets, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_flags: Bf16_flags, make_targets: Smoke_targets, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot( setup_flags: Fp16_flags, make_targets: Smoke_targets, needs_reboot:false, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_flags: Bf16_flags, make_targets: Smoke_targets, needs_reboot:false, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(lfs_pull: true,
                                                  setup_flags: "-DMIOPEN_TEST_DBSYNC=1",
                                                  make_targets: 'test_db_sync',
                                                  execute_cmd: './bin/test_db_sync',
                                                  needs_gpu:false,
                                                  needs_reboot:false,
                                                  build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(lfs_pull: true,
                                                  setup_flags: "-DMIOPEN_TEST_DBSYNC=1",
                                                  make_targets: 'test_db_sync',
                                                  execute_cmd: './bin/test_db_sync',
                                                  needs_gpu:false,
                                                  needs_reboot:false,
                                                  build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(lfs_pull: true,
                                                  setup_flags: "-DMIOPEN_TEST_DBSYNC=1",
                                                  make_targets: 'test_db_sync',
                                                  execute_cmd: './bin/test_db_sync',
                                                  needs_gpu:false,
                                                  needs_reboot:false,
                                                  build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot( setup_flags: Int8_flags + Full_test)
                        }
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
                            utils.buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: true)
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
                            utils.buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: true, needs_reboot:false)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_cmd: Navi21_build_cmd)
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
                            utils.buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags)
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
                            utils.buildHipClangJobAndReboot(setup_flags: Full_test)
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
                            utils.buildHipClangJobAndReboot(setup_flags: Full_test)
                        }
                    }
                }
                // stage('Fp32 Hip All gfx90a Xnack+') {
                //     when {
                //         beforeAgent true
                //         expression { params.TARGET_GFX90A && params.DATATYPE_FP32 }
                //     }
                //     agent{ label rocmnode("gfx90a") }
                //     steps{
                //        script {
                //         utils.buildHipClangJobAndReboot(setup_flags: Full_test, enforce_xnack_on: true)
                //        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_flags: Full_test, needs_reboot:false)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot( setup_flags: Full_test + Fp16_flags, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot( setup_flags: Full_test)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_flags: Full_test, build_cmd: Navi21_build_cmd, build_install: true)
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
                            utils.buildHipClangJobAndReboot(setup_flags: Full_test, build_install: true)
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
                            utils.buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_install: true)
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
                            utils.buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_install: true)
                        }
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
                        script {
                            utils.buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_install: true, needs_reboot:false)
                        }
                    }
                }
            }
        }
    }
}

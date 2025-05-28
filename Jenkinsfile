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
/// Target := { gfx908 | gfx90a | gfx94x } [ Xnack+ ]


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
            name: "TARGET_GFX94X",
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
            name: "WORKAROUND__TARGET_GFX94X_MINIMUM_TEST_ENABLE",
            defaultValue: false,
            description: "")
        booleanParam(
            name: "USE_SCCACHE_DOCKER",
            defaultValue: true,
            description: "Use the sccache for building CK in the Docker Image (default: ON)")
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
        stage("Package and Static checks") {
            when {
                expression { params.BUILD_PACKAGE_AND_CHECKS && params.TARGET_NOGPU && params.DATATYPE_NA }
            }
            parallel 
            {
                stage("HIP Package") {
                    agent{ label rocmnode("nogpu") }
                    steps{
                        script {
                            utils.buildHipClangJobAndReboot( package_build:true, needs_gpu:false, needs_reboot:false)
                        }
                    }
                }
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
                stage('Check GTest Format') {
                    agent { label rocmnode("nogpu") }
                    when {
                        changeset "**/test/gtest/**"
                    }
                    steps {
                        script {
                            checkout scm
                            sh 'cd ./test/utils && python3 gtest_formating_checks.py'
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
                stage('Tuna Fin Build Test') 
                {
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
        stage("DbSync") 
        {
            matrix {
                axes {
                    axis {
                        name 'DEVICE'
                        values 'gfx908', 'gfx90a', 'gfx94X'
                    }
                }
                stages {
                    stage("DbSync Tests") {
                        when {
                            beforeAgent true
                            allOf{
                                anyOf {
                                    expression { params.TARGET_GFX908 && "${DEVICE}" == 'gfx908' }
                                    expression { params.TARGET_GFX90A && "${DEVICE}" == 'gfx90a' }
                                    expression { params.TARGET_GFX94X && "${DEVICE}" == 'gfx94X' }
                                    //expression { params.TARGET_NAVI32 && "${DEVICE}" == 'gfx1101' }
                                }
                                expression { params.DBSYNC_TEST }
                            }
                        }
                        options {
                            retry(2)
                        }
                        agent{ label rocmnode("${DEVICE}") }
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
                }
            }
        }
        stage("Full Build & Test")
        {
            matrix {
                axes {
                    axis {
                        name 'DEVICE'
                        values 'gfx908', 'gfx90a', 'gfx94X'
                    }
                    axis {
                        name 'PRECISION'
                        values 'Fp32', 'Bf16', 'Fp16'
                    }
                }
                stages 
                {
                    stage("Hip Install All") 
                    {
                        when {
                            beforeAgent true
                            allOf{
                                anyOf {
                                    expression { params.TARGET_GFX908 && "${DEVICE}" == 'gfx908' }
                                    expression { params.TARGET_GFX90A && "${DEVICE}" == 'gfx90a' }
                                    expression { params.TARGET_GFX94X && "${DEVICE}" == 'gfx94X' }
                                }
                                anyOf {
                                    expression { params.DATATYPE_FP32 && "${PRECISION}" == 'Fp32' }
                                    expression { params.DATATYPE_BF16 && "${PRECISION}" == 'Bf16' }
                                    expression { params.DATATYPE_FP16 && "${PRECISION}" == 'Fp16' }
                                }
                            }
                        }
                        options {
                            retry(2)
                        }
                        agent{ label rocmnode("${DEVICE}") }
                        steps
                        {
                            script 
                            {
                                def flags = Full_test
                                if (PRECISION == 'Bf16') {
                                    flags += Bf16_flags
                                } else if (PRECISION == 'Fp16') {
                                    flags += Fp16_flags
                                }
                                utils.buildHipClangJobAndReboot(setup_flags: flags, build_install: true)
                            }
                        }
                    }
                }
            }
        }
        
    }
}

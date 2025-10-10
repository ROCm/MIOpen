/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
def miopenCheckout()
{
    // checkout project
    checkout([
        $class: 'GitSCM',
        branches: scm.branches,
        doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
        extensions: scm.extensions + [[$class: 'CleanCheckout']] + [[$class: 'SubmoduleOption', disableSubmodules: false, parentCredentials: true, recursiveSubmodules: true, ]],
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

def check_host() {
    if ("${env.MIOPEN_SCCACHE}" != "null"){
        def SCCACHE_SERVER="${env.MIOPEN_SCCACHE.split(':')[0]}"
        echo "sccache server: ${SCCACHE_SERVER}"
        sh '''ping -c 1 -p 6379 "${SCCACHE_SERVER}" | echo $? > tmp.txt'''
        def output = readFile(file: "tmp.txt")
        echo "tmp.txt contents: \$output"
        return (output != "0")
    }
    else{
        return 1
    }
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
    def miopen_install_path = conf.get("miopen_install_path", "${env.WORKSPACE}/${env.REPO_DIR}/install")

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

    def package_build = (conf.get("package_build",false) == true)

    if (package_build == true) {
        make_targets = "miopen_gtest package miopen_gtest_check"
        setup_args = " -DMIOPEN_TEST_DISCRETE=OFF " + setup_args
    }

    if(conf.get("build_install",false) == true)
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
        //grab root of node workspace. not guaranteed to be /var/jenkins
        String remote_root = env.WORKSPACE.substring(0, env.WORKSPACE.lastIndexOf("workspace/"))
        def vcache = conf.get(vcache_path,"${remote_root}/.cache/miopen/vcache")
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
            cd ${env.WORKSPACE}/${env.REPO_DIR}
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
            cd ${env.WORKSPACE}/${env.REPO_DIR}/fin
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
        params.PERF_TEST_ARCHIVE == true)) {
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
    sh "echo ${dockerArgs} > ${env.WORKSPACE}/factors.txt"
    def image = "${env.MIOPEN_DOCKER_IMAGE_URL}"
    sh "cd ${env.WORKSPACE}/${env.REPO_DIR}/ && md5sum Dockerfile requirements.txt dev-requirements.txt >> ${env.WORKSPACE}/factors.txt"
    def docker_hash = sh(script: "cd ${env.WORKSPACE} && md5sum factors.txt | awk '{print \$1}' | head -c 6", returnStdout: true)
    sh "rm ${env.WORKSPACE}/factors.txt"
    echo "Docker tag hash: ${docker_hash}"
    image = "${image}:ci_${docker_hash}"
    if(params.DOCKER_IMAGE_OVERRIDE && !params.DOCKER_IMAGE_OVERRIDE.empty)
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
    // Note: With offload compress disabled for CK expanding the target list might cause issues with the docker build.
    def gpu_arch = "gfx908;gfx90a;gfx942;gfx1151" // prebuilt dockers should have all the architectures enabled so one image can be used for all stages

    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 " +
                     "--build-arg PREFIX=${prefixpath} " +
                     "--build-arg GPU_ARCHS=\"${gpu_arch}\" "
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
    else if (params.USE_SCCACHE_DOCKER && check_host() && "${env.MIOPEN_SCCACHE}" != "null")
    {
        dockerArgs = dockerArgs + " --build-arg MIOPEN_SCCACHE=${env.MIOPEN_SCCACHE} --build-arg COMPILER_LAUNCHER=sccache"
    }
    echo "Docker Args: ${dockerArgs}"

    def image = getDockerImageName(dockerArgs)

    def dockerImage
    try{
        echo "Pulling down image: ${image}"
        dockerImage = docker.image("${image}")
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            dockerImage.pull()
        }
    }
    catch(org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
        echo "The job was cancelled or aborted"
        throw e
    }
    catch(Exception ex)
    {
        echo "Building image..."
        dockerImage = docker.build("${image}", "${dockerArgs} ${env.WORKSPACE}/${env.REPO_DIR}/.")
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            dockerImage.push()
        }
    }


    if(params.INSTALL_MIOPEN == 'ON')
    {
        def freckle = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
        dockerArgs = " --build-arg BASE_DOCKER=${image} --build-arg FRECKLE=${freckle} -f ${env.WORKSPACE}/${env.REPO_DIR}/Dockerfile.perftests"

        // Get updated image name for perf tests.
        image = getDockerImageName(dockerArgs)
        image = image + "_perfTest"

        try{
            echo "Pulling down perf test image: ${image}"
            dockerImage = docker.image("${image}")
            withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                dockerImage.pull()
            }
        }
        catch(org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
            echo "The job was cancelled or aborted"
            throw e
        }
        catch(Exception ex)
        {
            dockerImage = docker.build("${image}", "${dockerArgs} -f ${env.WORKSPACE}/${env.REPO_DIR}/.")
            withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                dockerImage.push()
            }
        }
    }

    return [dockerImage, image]
}

def buildHipClangJob(Map conf=[:]){
        show_node_info()
        /*
            The following is a workaround for git submodule updating for the fin module.  After Jenkins upgrade,
            many plugins started misbehaving, and submodules wouldn't get pulled.  This ensures that we always pull
            the fin submodule and fail silently when the submodule directory already has artifacts in it.
        */
        sh(script: "git submodule update --init --recursive || true")
        env.HSA_ENABLE_SDMA=0
        env.DOCKER_BUILDKIT=1
        def image
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1"
        }
        def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3')
        def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3')
        dockerOpts = dockerOpts + " --group-add=${video_id} --group-add=${render_id} "
        echo "Docker flags: ${dockerOpts}"

        def variant = env.STAGE_NAME

        def needs_gpu = conf.get("needs_gpu", true)
        def dvc_pull = conf.get("dvc_pull", false)
        def build_timeout = conf.get("build_timeout", 420)

        def retimage
        def credentialsID = env.monorepo_status_wrapper_creds
        if (env.REPO_NAME == "MIOpen") {
            credentialsID = env.miopen_git_creds
        }
        gitStatusWrapper(credentialsId: "${credentialsID}", gitHubContext: "${variant}", account: 'ROCm', repo: "${env.REPO_NAME}") {
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

            //grab root of node workspace. not guaranteed to be /var/jenkins
            String remote_root = env.WORKSPACE.substring(0, env.WORKSPACE.lastIndexOf("workspace/"))
            withDockerContainer(image: image, args: dockerOpts + " -v=${remote_root}:${remote_root}") {
                timeout(time: build_timeout, unit:'MINUTES')
                {
                    // We set LOGNAME here because under the hood dvc calls Python's getpass.getuser() object to 
                    // create a unique hash to store its local cache in. When Jenkins runs this Docker container, it
                    // runs as a UID that doesn't have an entry in /etc/passwd within the container. getuser() throws
                    // an exception if it can't get the user name for the current user's UID, but it will check the 
                    // LOGNAME environment variable and use that value if it's available.
                    // https://github.com/iterative/dvc/blob/3915fa26aa7d95d5cbe345e62846bfd82dccbfc7/dvc/repo/__init__.py#L646
                    // https://docs.python.org/3/library/getpass.html#getpass.getuser
                    if (dvc_pull) {
                        sh """
                            cd ${env.WORKSPACE}/${env.REPO_DIR}
                            LOGNAME=temp-user dvc pull -v
                           """.stripIndent()
                    }
                    cmake_build(conf)
                }
            }
        }
        return retimage
}

def RunPerfTest(Map conf=[:]){
    def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    try {
        def docker_image = conf.get("docker_image")
        def miopen_install_path = conf.get("miopen_install_path", "/opt/rocm")
        def results_dir = conf.get("results_dir", "${env.WORKSPACE}/${env.REPO_DIR}/results")
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            docker_image.pull()
        }
        echo "docker image: ${docker_image}"
        //grab root of node workspace. not guaranteed to be /var/jenkins
        String remote_root = env.WORKSPACE.substring(0, env.WORKSPACE.lastIndexOf("workspace/"))
        docker_image.inside(dockerOpts + " -v=${remote_root}:${remote_root}")
        {
            timeout(time: 100, unit: 'MINUTES')
            {
                ld_lib="${miopen_install_path}/lib"
                def filename = conf.get("filename")
                assert(filename.trim())
                def cmd = "export LD_LIBRARY_PATH=${ld_lib} && ${miopen_install_path}/share/miopen/bin/test_perf.py  --filename ${filename} --install_path ${miopen_install_path} --results_path ${results_dir}/perf_results"
                if(params.PERF_TEST_OVERRIDE != '')
                {
                    echo "Appending MIOpenDriver cmd env vars: ${params.PERF_TEST_OVERRIDE}"
                    cmd += " --override ${params.PERF_TEST_OVERRIDE}"
                }
                sh cmd
                archiveArtifacts artifacts: "results/perf_results/${filename}", allowEmptyArchive: true, fingerprint: true
                jenkins_url = "${env.artifact_path}/${env.JOB_BASE_NAME}/lastSuccessfulBuild/artifact/results/perf_results"
                if(params.COMPARE_TO_BASE)
                {
                  try {
                      sh "rm -rf ${results_dir}/old_results/"
                      sh "wget -P ${results_dir}/old_results/ ${jenkins_url}/${filename}"
                  }
                  catch (Exception err){
                      currentBuild.result = 'SUCCESS'
                  }

                  try{
                     sh "${miopen_install_path}/share/miopen/bin/test_perf.py --compare_results --old_results_path ${results_dir}/old_results --results_path ${results_dir}/perf_results --filename ${filename}"
                  }
                  catch (Exception err){
                      currentBuild.result = 'SUCCESS'
                  }
                  cleanWs()
                }
            }
        }
    }
    catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
        echo "The job was cancelled or aborted"
        throw e
    }
}

return this

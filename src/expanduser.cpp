/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/stringutils.hpp>

#include <boost/filesystem.hpp>

#include <string>

#ifdef __linux__
#include <errno.h>
#include <string.h>
#include <sys/vfs.h>
#include <linux/magic.h>

#ifndef LL_SUPER_MAGIC
#define LL_SUPER_MAGIC \
    0x0BD00BD0 // LUSTRE
               // https://github.com/whamcloud/lustre/blob/a336d7c7c1cd62a5a5213835aa85b8eaa87b076a/lustre/include/uapi/linux/lustre/lustre_user.h#L252
#endif
#ifndef CEPH_SUPER_MAGIC
#define CEPH_SUPER_MAGIC 0x00c36400
#endif
#ifndef NFS_SUPER_MAGIC
#define NFS_SUPER_MAGIC 0x6969
#endif
#ifndef SMB_SUPER_MAGIC
#define SMB_SUPER_MAGIC 0x517b
#endif
#ifndef SMB2_MAGIC_NUMBER
#define SMB2_MAGIC_NUMBER 0xfe534d42
#endif
#ifndef CIFS_MAGIC_NUMBER
#define CIFS_MAGIC_NUMBER 0xff534d42
#endif
#ifndef CODA_SUPER_MAGIC
#define CODA_SUPER_MAGIC 0x73757245
#endif
#ifndef OCFS2_SUPER_MAGIC
#define OCFS2_SUPER_MAGIC 0x7461636f
#endif
#ifndef AFS_SUPER_MAGIC
#define AFS_SUPER_MAGIC 0x5346414f
#endif
#ifndef EXT2_OLD_SUPER_MAGIC
#define EXT2_OLD_SUPER_MAGIC 0xef51
#endif
#ifndef EXT4_SUPER_MAGIC
#define EXT4_SUPER_MAGIC 0xef53
#endif
#ifndef TMPFS_MAGIC
#define TMPFS_MAGIC 0x01021994
#endif
#ifndef OVERLAYFS_SUPER_MAGIC
#define OVERLAYFS_SUPER_MAGIC 0x794c7630
#endif
#endif // __linux__

MIOPEN_DECLARE_ENV_VAR(HOME)

namespace miopen {

#ifdef __linux__

#define CASE_RET_STRING(macro) \
    case macro: return #macro;

namespace {
const char* Stringize(unsigned long ft)
{
    switch(ft)
    {
        CASE_RET_STRING(NFS_SUPER_MAGIC)
        CASE_RET_STRING(SMB_SUPER_MAGIC)
        CASE_RET_STRING(SMB2_MAGIC_NUMBER)
        CASE_RET_STRING(CIFS_MAGIC_NUMBER)
        CASE_RET_STRING(CODA_SUPER_MAGIC)
        CASE_RET_STRING(OCFS2_SUPER_MAGIC)
        CASE_RET_STRING(AFS_SUPER_MAGIC)
        CASE_RET_STRING(LL_SUPER_MAGIC)
        CASE_RET_STRING(CEPH_SUPER_MAGIC)
        CASE_RET_STRING(TMPFS_MAGIC)
        CASE_RET_STRING(OVERLAYFS_SUPER_MAGIC)
        CASE_RET_STRING(EXT2_OLD_SUPER_MAGIC)
    case EXT4_SUPER_MAGIC: return "EXT2/3/4_SUPER_MAGIC";
    default: return "<Unknown magic>";
    }
}

bool IsNetworked(unsigned long ft)
{
    switch(ft)
    {
    case NFS_SUPER_MAGIC:   // fall through
    case SMB_SUPER_MAGIC:   // fall through
    case SMB2_MAGIC_NUMBER: // fall through
    case CIFS_MAGIC_NUMBER: // fall through
    case CODA_SUPER_MAGIC:  // fall through
    case OCFS2_SUPER_MAGIC: // fall through
    case AFS_SUPER_MAGIC:   // fall through
    case LL_SUPER_MAGIC:    // fall through
    case CEPH_SUPER_MAGIC: return true;
    default: return false;
    }
}
} // namespace

#undef CASE_RET_STRING
#endif // __linux__

bool IsNetworkedFilesystem(const boost::filesystem::path& path)
{
#ifdef __linux__
    struct statfs stat;
    const int rc = statfs(path.c_str(), &stat);
    if(rc != 0)
    {
        MIOPEN_LOG_NQE("rc = " << rc << ", '" << strerror(errno) << "'");
        return false;
    }
    MIOPEN_LOG_NQI("Filesystem type at '" << path.string() << "' is: 0x" << std::hex << stat.f_type
                                          << " '" << Stringize(stat.f_type) << '\'');
    return IsNetworked(stat.f_type);
#else
    std::ignore = path;
    return false;
#endif // __linux__
}

namespace {
std::string GetHomeDir()
{
    const char* const p = GetStringEnv(HOME{});
    if(!(p == nullptr || p == std::string("/") || p == std::string("")))
    {
        return {p};
    }
    // todo:
    // need to figure out what is the correct thing to do here
    // in tensoflow unit tests run via bazel, $HOME is not set, so this can happen
    // setting home_dir to the /tmp for now
    return {boost::filesystem::temp_directory_path().string()};
}
} // namespace

boost::filesystem::path ExpandUser(const std::string& path)
{
    static const std::string home_dir = GetHomeDir();
    return {ReplaceString(path, "~", home_dir)};
}

} // namespace miopen

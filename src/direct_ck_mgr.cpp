#include "miopen/direct_ck_mgr.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <sstream>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>

std::unordered_map<size_t, CacheData> DirectCkMgr::s_qun_wrw =
{
    {0x1d8e031b28fe3461, {0x1d8e031b28fe3461, 0x8ada6ba6c0a8262f, 0x20}},
    {0x0e92b755dc1549cc, {0x0e92b755dc1549cc, 0x9edc399f25f4e3cf, 0x1}},
    {0x37896c980651dd6a, {0x37896c980651dd6a, 0x3b6498916e34a84c, 0x4}},
    {0x25d1a9f7bf77cdb1, {0x25d1a9f7bf77cdb1, 0x9edc399f25f4e3cf, 0x1}},
    {0x2dcf0679e65a2176, {0x2dcf0679e65a2176, 0xb2b47d69d3efff96, 0x8}},
    {0x7cda3462205873e9, {0x7cda3462205873e9, 0x8caaa26caf0cf637, 0x4}},
    {0x4b30467a1ed12e94, {0x4b30467a1ed12e94, 0x634867055bc5ccb3, 0x10}},
    {0xe51eab72929667c7, {0xe51eab72929667c7, 0x7cdccadfae7e4912, 0x20}},
    {0x862a553bc1b6135f, {0x862a553bc1b6135f, 0x7cdccadfae7e4912, 0x20}},
    {0xc6075c461b1f4387, {0xc6075c461b1f4387, 0x18c034c0dd790bc9, 0x1}},
    {0xfcaf93b538f3b837, {0xfcaf93b538f3b837, 0x55335f15571bd2d2, 0x1}},
    {0xce0513f63f76d7e2, {0xce0513f63f76d7e2, 0xb62b34a2c839689b, 0x1}},
    {0x6de0f5c2e31130ad, {0x6de0f5c2e31130ad, 0x6b529345f00e682f, 0x1}},
    {0x7fd11c3a9a196bdb, {0x7fd11c3a9a196bdb, 0xd8cef6f43ec57670, 0x20}},
};

std::unordered_map<size_t, CacheData> DirectCkMgr::s_qun_fwd = {
    {0x255b55cff8f3c0e4, {0x255b55cff8f3c0e4, 0xd7cd1611c2b102fd, 0x1}},
    {0xd1de99352cbf1d84, {0xd1de99352cbf1d84, 0xb2e97955866cbab4, 0x1}},
    {0x3277e944844d2c62, {0x3277e944844d2c62, 0x0670ac3fb50f4554, 0x1}},
    {0xdc6fb0164d9beac2, {0xdc6fb0164d9beac2, 0xb2e97955866cbab4, 0x1}},
    {0xc6382b70b4607bea, {0xc6382b70b4607bea, 0x8538e4616d752975, 0x1}},
    {0x38c36eaf439cfe27, {0x38c36eaf439cfe27, 0x0670ac3fb50f4554, 0x1}},
    {0x40640df67da3b570, {0x40640df67da3b570, 0x61a58d613313e6ad, 0x1}},
    {0xe40d59a8bfd9e90b, {0xe40d59a8bfd9e90b, 0x49e7832040894571, 0x1}},
    {0x3cc0ab518d41860e, {0x3cc0ab518d41860e, 0x1397dc786799a32f, 0x1}},
    {0xe0a79ed390315ef3, {0xe0a79ed390315ef3, 0xd866939484178a07, 0x1}},
    {0xc01921a2c5162725, {0xc01921a2c5162725, 0xd225957a95caeee5, 0x1}},
    {0xe61eba76c9736b03, {0xe61eba76c9736b03, 0xd225957a95caeee5, 0x1}},
    {0xf28b9ebd2c8f8500, {0xf28b9ebd2c8f8500, 0x75d438f1e790f9a1, 0x1}},
    {0xb1e77d2102144586, {0xb1e77d2102144586, 0xc6d91923648393aa, 0x1}},
};

std::unordered_map<size_t, CacheData> DirectCkMgr::s_jin_bwd = {
    {0x22e622671bdb6c09, {0x22e622671bdb6c09, 0xc63a4d7d2c1ae55b, 0x1}},
    {0x41f5b2a76e864672, {0x41f5b2a76e864672, 0x923bc05ca98b2f39, 0x1}},
    {0xe4017519878aa942, {0xe4017519878aa942, 0x3739bad3a3a16ce0, 0x1}},
    {0xcbb2b84def46d737, {0xcbb2b84def46d737, 0x923bc05ca98b2f39, 0x1}},
    {0x8e3decf13e67cc3b, {0x8e3decf13e67cc3b, 0x986cef20b781a571, 0x1}},
    {0xb3c139e2635ef3a1, {0xb3c139e2635ef3a1, 0xa783d8df4049bc8e, 0x1}},
    {0x048fdb43442ae17e, {0x048fdb43442ae17e, 0xaf6b0ec63cb6d4f6, 0x1}},
    {0x38971f660cb6ad37, {0x38971f660cb6ad37, 0x5b65e93e5f11971b, 0x1}},
    {0x25d4b080aea13376, {0x25d4b080aea13376, 0x473082db002e3652, 0x1}},
    {0x4c9bcaea4dd91d63, {0x4c9bcaea4dd91d63, 0x5b65e93e5f11971b, 0x1}},
    {0x20b319089592b37c, {0x20b319089592b37c, 0x473082db002e3652, 0x1}},
    {0x57d7d22c2a7d4e8a, {0x57d7d22c2a7d4e8a, 0x31e11a4016e48530, 0x1}},
    {0xe1bf8dd31f895744, {0xe1bf8dd31f895744, 0xfa059fdad40b15b4, 0x1}},
    {0x76b3bbe8d6e84721, {0x76b3bbe8d6e84721, 0xb02d7e1ce8e73e0a, 0x1}},
};

static bool IsEnvEnabled(const char* pEnv, bool defVal)
{
    const char* env_val = std::getenv(pEnv);
    if (env_val == nullptr)
    {
        return defVal;
    }

    std::string value(env_val);
    
    // Trim leading whitespace
    size_t start = value.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return false;
    }
    value.erase(0, start);

    // Trim trailing whitespace
    size_t end = value.find_last_not_of(" \t\n\r\f\v");
    if (end != std::string::npos) {
        value.erase(end + 1);
    }

    return value == "1";
}

DirectCkMgr::DirectCkMgr()
{
    Init();
}

void DirectCkMgr::Init()
{
    enableConvCache = IsEnvEnabled("DCK_CONV_FILE_CACHE", enableConvCache);
    enableOptConv   = IsEnvEnabled("DCK_OPT_CONV", enableOptConv);
    enableLog       = IsEnvEnabled("DCK_LOG_ENABLE", enableLog);

    std::string home_path = std::filesystem::path(std::getenv("HOME"));
    path_qun_wrw = home_path / std::filesystem::path(".config/miopen/dck_conv_qun_wrw.txt");

    path_qun_fwd = home_path / std::filesystem::path(".config/miopen/dck_conv_qun_fwd.txt");

    path_jin_bwd = home_path / std::filesystem::path(".config/miopen/dck_conv_jin_bwd.txt");

    std::filesystem::create_directories(path_qun_wrw.parent_path());
}

size_t DirectCkMgr::GetStringHash(std::string str)
{
    std::hash<std::string> hasher;
    size_t hashValue = hasher(str);

    return hashValue;
}

void DirectCkMgr::FlushToCacheFile(std::unordered_map<size_t, CacheData>& input, const char* pPath)
{
    if (enableConvCache == false)   return;

    int fd = open(pPath, O_RDWR | O_CREAT, 0666);
    if (fd != -1)
    {
        if (flock(fd, LOCK_EX) == -1)
        {
            std::cerr << "Error locking the file." << std::endl;
            close(fd);
            return;
        }

        // Read existing
        std::ifstream infile(pPath);
        std::unordered_map<size_t, CacheData> existingData;
        if (infile)
        {
            std::string line;
            while (std::getline(infile, line))
            {
                size_t hashcode, kernalHash;
                int split_k;
                std::istringstream iss(line);
                if (!(iss >> std::hex >> hashcode >> std::hex >> kernalHash >> split_k))
                {
                    continue;
                }

                CacheData cd = { hashcode, kernalHash, split_k };

                if (existingData.find(cd.hashcode) == existingData.end())
                {
                    existingData[hashcode] = cd;
                }
            }

            infile.close();
        }

        std::ofstream outfile(pPath, std::ios::app);

        if (outfile)
        {
            for (const auto& [key, cd] : input)
            {
                if (existingData.find(key) == existingData.end())
                {
                    outfile << std::hex << std::setw(16) << std::setfill('0') << cd.hashcode << " "
                            << std::hex << std::setw(16) << std::setfill('0') << cd.kernelhash << " "
                            << cd.split_k << "\n";
                }
            }

            outfile.close();
        }

        flock(fd, LOCK_UN);
        close(fd);
    }
}

DirectCkMgr::~DirectCkMgr()
{
    if (newKernelCount[ST_QUN_WRW] > 0)
        FlushToCacheFile(s_qun_wrw, path_qun_wrw.c_str());
    if (newKernelCount[ST_QUN_FWD] > 0)
        FlushToCacheFile(s_qun_fwd, path_qun_fwd.c_str());
    if (newKernelCount[ST_JIN_BWD] > 0)
        FlushToCacheFile(s_jin_bwd, path_jin_bwd.c_str());

    // DEBUG LOG
    if (enableLog)
    {
        std::cout <<"Name\t hitCount\t launchCount:" << std::endl;
        if (launchCount[ST_QUN_WRW] != 0)
            std::cout <<"qun wrw: " << hitCacheCount[ST_QUN_WRW] <<"\t "<< launchCount[ST_QUN_WRW] << std::endl;
        if (launchCount[ST_QUN_FWD] != 0)
            std::cout <<"qun fwd: " << hitCacheCount[ST_QUN_FWD] <<"\t "<< launchCount[ST_QUN_FWD] << std::endl;
        if (launchCount[ST_JIN_BWD] != 0)
            std::cout <<"jin bwd: " << hitCacheCount[ST_JIN_BWD] <<"\t "<< launchCount[ST_JIN_BWD] << std::endl;
    }

}

void DirectCkMgr::ReadCacheFile(std::unordered_map<size_t, CacheData>& outputData, std::filesystem::path filePath)
{
    std::ifstream infile(filePath);
    if (infile)
    {
        std::string line;
        while (std::getline(infile, line))
        {
            size_t hashcode, kernalHash, split_k;
            std::istringstream iss(line);
            if (!(iss >> std::hex >> hashcode >> std::hex >> kernalHash >> std::hex >> split_k))
            {
                continue;
            }

            CacheData cd = { hashcode, kernalHash, 1 };
            if (outputData.find(cd.hashcode) == outputData.end())
            {
                outputData[hashcode] = cd;
            }
        }

        infile.close();
    }
}

void DirectCkMgr::AppendToCache(std::unordered_map<size_t, CacheData>& outputData, CacheData cd)
{
    if (outputData.find(cd.hashcode) == outputData.end())
    {
        outputData[cd.hashcode] = cd;
    }
}

bool DirectCkMgr::FindCacheData(std::unordered_map<size_t, CacheData>& inputData, size_t hashcode, CacheData& cd)
{
    bool found = false;
    {
        auto it = inputData.find(hashcode);
        found = it != inputData.end();
        if (found)
        {
            cd = it->second;
        }
    }

    return found;
}

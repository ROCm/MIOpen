
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

#include <miopen/lock_file.hpp>
#include <gtest/gtest.h>

template <class TDuration>
static boost::posix_time::ptime ToPTime(TDuration duration)
{
    return boost::posix_time::second_clock::universal_time() +
           boost::posix_time::milliseconds(
               std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
}

TEST(CPU_UnitTestLockFile_NONE, TryLock)
{
    miopen::fs::remove_all(miopen::fs::path{"/tmp/config/miopen"});
    auto lockpath = miopen::LockFilePath(miopen::fs::path{"/tmp/config/miopen/test"});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";
    auto lockfile                 = miopen::FSLockFile(lockpath);
    auto unique_handle            = lockfile.get_unique_handle();

    EXPECT_FALSE(miopen::fs::exists(unique_handle));
    EXPECT_TRUE(lockfile.try_lock());
    EXPECT_TRUE(miopen::fs::exists(unique_handle));
    EXPECT_TRUE(miopen::fs::hard_link_count(unique_handle) == 2);

    EXPECT_FALSE(lockfile.try_lock());
    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
    lockfile.unlock();
    EXPECT_FALSE(miopen::fs::exists(unique_handle));
}

TEST(CPU_UnitTestLockFile_NONE, TimedLock)
{
    miopen::fs::remove_all(miopen::fs::path{"/tmp/config/miopen"});
    auto lockpath = miopen::LockFilePath(miopen::fs::path{"/tmp/config/miopen/test"});
    auto lockfile = miopen::FSLockFile(lockpath);
    EXPECT_TRUE(lockfile.timed_lock(ToPTime(std::chrono::seconds{1})));
    EXPECT_FALSE(lockfile.timed_lock(ToPTime(std::chrono::seconds{1})));
    auto unique_handle = lockfile.get_unique_handle();
    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, OrphanLock)
{
    miopen::fs::remove_all(miopen::fs::path{"/tmp/config/miopen"});
    auto lockpath = miopen::LockFilePath(miopen::fs::path{"/tmp/config/miopen/test"});
    auto lockfile = miopen::FSLockFile(lockpath);
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";
    auto unique_handle            = lockfile.get_unique_handle();

    EXPECT_TRUE(std::ofstream{fs_lock_path});
    miopen::fs::permissions(fs_lock_path, miopen::fs::perms::all);

    miopen::fs::last_write_time(fs_lock_path, miopen::fs::file_time_type::clock::now());
    EXPECT_FALSE(lockfile.try_lock());
    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
    EXPECT_TRUE(lockfile.timed_lock(ToPTime(std::chrono::milliseconds{50})));

    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, LostLockFile)
{
    miopen::fs::remove_all(miopen::fs::path{"/tmp/config/miopen"});
    auto lockpath = miopen::LockFilePath(miopen::fs::path{"/tmp/config/miopen/test"});
    auto lockfile = miopen::FSLockFile(lockpath);
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";
    auto unique_handle            = lockfile.get_unique_handle();

    EXPECT_TRUE(lockfile.try_lock());
    miopen::fs::remove(unique_handle);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, RefreshHold)
{
    miopen::fs::remove_all(miopen::fs::path{"/tmp/config/miopen"});
    auto lockpath = miopen::LockFilePath(miopen::fs::path{"/tmp/config/miopen/test"});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";
    auto lockfile                 = miopen::FSLockFile(lockpath);
    auto unique_handle            = lockfile.get_unique_handle();

    EXPECT_TRUE(lockfile.try_lock());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(lockfile.try_lock());
    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));

    lockfile.unlock();
    EXPECT_TRUE(lockfile.try_lock());
    lockfile.unlock();
}

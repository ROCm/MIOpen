/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/errors.hpp>
#include <miopen/lock_file.hpp>
#include <miopen/logger.hpp>
#include <miopen/filesystem.hpp>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/none.hpp>
#include <boost/optional.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <ios>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace miopen {

PlainTextDb::PlainTextDb(DbKinds db_kind_, const fs::path& filename_, bool is_system)
    : db_kind(db_kind_),
      filename(filename_),
      lock_file(LockFile::Get(LockFilePath(filename_))),
      warning_if_unreadable(is_system)
{
    if(is_system)
    {
        MIOPEN_THROW("PlainTextDb class is not supported as system database. Use the ReadOnlyRamDb "
                     "class instead.");
    }

    if(!DisableUserDbFileIO)
    {
        fs::path directory = filename.has_parent_path() ? filename.parent_path() : "";
        if(!fs::exists(directory))
        {
            if(!fs::create_directories(directory))
                MIOPEN_LOG_W("Unable to create a directory: " << directory);
            else
                fs::permissions(directory, FS_ENUM_PERMS_ALL);
        }
    }
}

#define MIOPEN_VALIDATE_LOCK(lock)                       \
    do                                                   \
    {                                                    \
        if(!(lock))                                      \
            MIOPEN_THROW("Db lock has failed to lock."); \
    } while(false)

static std::chrono::seconds GetLockTimeout() { return std::chrono::seconds{60}; }

using exclusive_lock = std::unique_lock<LockFile>;
using shared_lock    = std::shared_lock<LockFile>;

boost::optional<DbRecord> PlainTextDb::FindRecord(const std::string& key)
{
    if(DisableUserDbFileIO)
        return {};
    const auto lock = shared_lock(lock_file, GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    return FindRecordUnsafe(key, nullptr);
}

bool PlainTextDb::StoreRecord(const DbRecord& record)
{
    if(DisableUserDbFileIO)
        return true;
    const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    return StoreRecordUnsafe(record);
}

bool PlainTextDb::UpdateRecord(DbRecord& record)
{
    if(DisableUserDbFileIO)
        return true;
    const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    return UpdateRecordUnsafe(record);
}

bool PlainTextDb::RemoveRecord(const std::string& key)
{
    if(DisableUserDbFileIO)
        return true;
    const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    return RemoveRecordUnsafe(key);
}

bool PlainTextDb::Remove(const std::string& key, const std::string& id)
{
    if(DisableUserDbFileIO)
        return true;
    const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    auto record = FindRecordUnsafe(key, nullptr);
    if(!record)
        return false;
    bool erased = record->EraseValues(id);
    if(!erased)
        return false;
    return StoreRecordUnsafe(*record);
}

boost::optional<DbRecord> PlainTextDb::FindRecordUnsafe(const std::string& key,
                                                        RecordPositions* pos)
{
    if(pos != nullptr)
    {
        pos->begin = -1;
        pos->end   = -1;
    }

    MIOPEN_LOG_I2("Looking for key " << key << " in file " << filename);

    std::ifstream file(filename, std::ios::binary);

    if(!file)
    {
        const auto log_level = IsWarningIfUnreadable() && !MIOPEN_DISABLE_SYSDB
                                   ? LoggingLevel::Warning
                                   : LoggingLevel::Info2;
        MIOPEN_LOG(log_level, "File is unreadable: " << filename);
        return boost::none;
    }

    int n_line = 0;
    while(true)
    {
        std::string line;
        const auto line_begin = file.tellg();
        if(!std::getline(file, line))
            break;
        ++n_line;
        const auto next_line_begin = file.tellg();

        const auto key_size = line.find('=');
        const bool is_key   = (key_size != std::string::npos && key_size != 0);
        if(!is_key)
        {
            if(!line.empty()) // Do not blame empty lines.
            {
                MIOPEN_LOG_E("Ill-formed record: key not found: " << filename << "#" << n_line);
            }
            continue;
        }
        const auto current_key = line.substr(0, key_size);

        if(current_key != key)
        {
            continue;
        }
        MIOPEN_LOG_I2("Key match: " << current_key);
        const auto contents = line.substr(key_size + 1);

        if(contents.empty())
        {
            MIOPEN_LOG_E("None contents under the key: " << current_key << " form file " << filename
                                                         << "#" << n_line);
            continue;
        }
        MIOPEN_LOG_I2("Contents found: " << contents);

        DbRecord record(key);
        const bool is_parse_ok = record.ParseContents(contents);

        if(!is_parse_ok)
        {
            MIOPEN_LOG_E("Error parsing payload under the key: " << current_key << " form file "
                                                                 << filename << "#" << n_line);
            MIOPEN_LOG_E("Contents: " << contents);
        }
        // A record with matching key have been found.
        if(pos != nullptr)
        {
            pos->begin = line_begin;
            pos->end   = next_line_begin;
        }
        return record;
    }
    // Record was not found
    return boost::none;
}

static void Copy(std::istream& from, std::ostream& to, std::streamoff count)
{
    constexpr auto buffer_size_limit = 4 * 1024 * 1024;
    const auto buffer_size           = std::min<std::streamoff>(buffer_size_limit, count);
    auto buffer                      = std::vector<char>(buffer_size, 0);
    auto left                        = count;

    while(left > 0 && !from.eof())
    {
        const auto to_read = std::min<std::streamoff>(left, buffer_size);
        from.read(buffer.data(), to_read);
        const auto read = from.gcount();
        to.write(buffer.data(), read);
        left -= read;
    }
}

bool PlainTextDb::FlushUnsafe(const DbRecord& record, const RecordPositions* pos)
{
    assert(pos);

    if(pos->begin < 0 || pos->end < 0)
    {
        {
            std::ofstream file(filename, std::ios::app | std::ios::binary);

            if(!file)
            {
                MIOPEN_LOG_E("File is unwritable: " << filename);
                return false;
            }

            (void)file.tellp();
            record.WriteContents(file);
        }

        fs::permissions(filename, FS_ENUM_PERMS_ALL);
    }
    else
    {
        std::ifstream from(filename, std::ios::ate | std::ios::binary);

        if(!from)
        {
            MIOPEN_LOG_E("File is unreadable: " << filename);
            return false;
        }

        const auto temp_name = filename + ".temp";
        std::ofstream to(temp_name, std::ios::binary);

        if(!to)
        {
            MIOPEN_LOG_E("Temp file is unwritable: " << temp_name);
            return false;
        }

        const auto from_size = from.tellg();
        from.seekg(std::ios::beg);

        Copy(from, to, pos->begin);
        record.WriteContents(to);
        from.seekg(pos->end);
        Copy(from, to, from_size - pos->end);

        from.close();
        to.close();

        fs::remove(filename);
        fs::rename(temp_name, filename);
        /// \todo What if rename fails? Thou shalt not loose the original file.
        fs::permissions(filename, FS_ENUM_PERMS_ALL);
    }
    return true;
}

bool PlainTextDb::StoreRecordUnsafe(const DbRecord& record)
{
    MIOPEN_LOG_I2("Storing record: " << record.key);
    RecordPositions pos;
    FindRecordUnsafe(record.key, &pos);
    return FlushUnsafe(record, &pos);
}

bool PlainTextDb::UpdateRecordUnsafe(DbRecord& record)
{
    RecordPositions pos;
    const auto old_record = FindRecordUnsafe(record.key, &pos);
    DbRecord new_record(record);
    if(old_record)
    {
        new_record.Merge(*old_record);
        MIOPEN_LOG_I2("Updating record: " << record.key);
    }
    else
    {
        MIOPEN_LOG_I2("Storing record: " << record.key);
    }
    bool result = FlushUnsafe(new_record, &pos);
    if(result)
        record = std::move(new_record);
    return result;
}

bool PlainTextDb::RemoveRecordUnsafe(const std::string& key)
{
    // Create empty record with same key and replace original with that
    // This will remove record
    MIOPEN_LOG_I("Removing record: " << key);
    RecordPositions pos;
    FindRecordUnsafe(key, &pos);
    const DbRecord empty_record(key);
    return FlushUnsafe(empty_record, &pos);
}

} // namespace miopen

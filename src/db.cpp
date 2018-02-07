#include <miopen/db.hpp>

namespace miopen {

boost::optional<DbRecord> Db::FindRecordUnsafe(const std::string& key, RecordPositions* pos)
{
    if(pos)
    {
        pos->begin = -1;
        pos->end   = -1;
    }

    MIOPEN_LOG_I("Looking for key: " << key);

    std::ifstream file(filename);

    if(!file)
    {
        MIOPEN_LOG_W("File is unreadable.");
        return boost::none;
    }

    int n_line = 0;
    while(true)
    {
        std::string line;
        std::streamoff line_begin = file.tellg();
        if(!std::getline(file, line))
            break;
        ++n_line;
        std::streamoff next_line_begin = file.tellg();

        const auto key_size = line.find('=');
        const bool is_key   = (key_size != std::string::npos && key_size != 0);
        if(!is_key)
        {
            if(!line.empty()) // Do not blame empty lines.
            {
                MIOPEN_LOG_E("Ill-formed record: key not found.");
                MIOPEN_LOG_E(filename << "#" << n_line);
            }
            continue;
        }
        const auto current_key = line.substr(0, key_size);

        if(current_key != key)
        {
            continue;
        }
        MIOPEN_LOG_I("Key match: " << current_key);
        const auto contents = line.substr(key_size + 1);

        if(contents.empty())
        {
            MIOPEN_LOG_E("None contents under the key: " << current_key);
            continue;
        }
        MIOPEN_LOG_I("Contents found: " << contents);

        DbRecord record(key);
        const bool is_parse_ok = record.ParseContents(contents);

        if(!is_parse_ok)
        {
            MIOPEN_LOG_E("Error parsing payload under the key: " << current_key);
            MIOPEN_LOG_E(filename << "#" << n_line);
            MIOPEN_LOG_E(contents);
        }
        // A record with matching key have been found.
        if(pos)
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
    constexpr auto buffer_size = 4 * 1024 * 1024;
    char buffer[buffer_size];
    auto left = count;

    while(left > 0 && !from.eof())
    {
        const auto to_read = std::min<std::streamoff>(left, buffer_size);
        from.read(buffer, to_read);
        const auto read = from.gcount();
        to.write(buffer, read);
        left -= read;
    }
}

bool Db::FlushUnsafe(const DbRecord& record, const RecordPositions* pos)
{
    assert(pos);

    if(pos->begin < 0 || pos->end < 0)
    {
        std::ofstream file(filename, std::ios::app);

        if(!file)
        {
            MIOPEN_LOG_E("File is unwritable.");
            return false;
        }

        (void)file.tellp();
        record.WriteContents(file);
    }
    else
    {
        const auto temp_name = filename + ".temp";
        std::ifstream from(filename, std::ios::ate);

        if(!from)
        {
            MIOPEN_LOG_E("File is unreadable.");
            return false;
        }

        std::ofstream to(temp_name);

        if(!to)
        {
            MIOPEN_LOG_E("Temp file is unwritable.");
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

        std::remove(filename.c_str());
        std::rename(temp_name.c_str(), filename.c_str());
        /// \todo What if rename fails? Thou shalt not loose the original file.
    }
    return true;
}

bool Db::StoreRecordUnsafe(const DbRecord& record)
{
    MIOPEN_LOG_I("Storing record: " << record.key);
    RecordPositions pos;
    auto old_record = FindRecordUnsafe(record.key, &pos);
    return FlushUnsafe(record, &pos);
}

bool Db::UpdateRecordUnsafe(DbRecord& record)
{
    RecordPositions pos;
    auto old_record = FindRecordUnsafe(record.key, &pos);
    DbRecord new_record(record);
    if(old_record)
    {
        new_record.Merge(*old_record);
        MIOPEN_LOG_I("Updating record: " << record.key);
    }
    else
    {
        MIOPEN_LOG_I("Storing record: " << record.key);
    }
    bool result = FlushUnsafe(new_record, &pos);
    if(result)
        record = std::move(new_record);
    return result;
}

bool Db::RemoveRecordUnsafe(const std::string& key)
{
    // Create empty record with same key and replace original with that
    // This will remove record
    MIOPEN_LOG_I("Removing record: " << key);
    RecordPositions pos;
    FindRecordUnsafe(key, &pos);
    DbRecord empty_record(key);
    return FlushUnsafe(empty_record, &pos);
}

} // namespace miopen
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <algorithm>

void Bin2Hex(std::istream& source, std::ostream& target, const std::string& variable, bool nullTerminate, size_t bufferSize, size_t lineSize)
{
	std::unique_ptr<unsigned char[]> buffer(new unsigned char[bufferSize]);
	std::streamoff sourceSize = source.tellg();
	std::streamoff blockStart = 0;

	if (variable.length() != 0)
	{
		target << "const size_t " << variable << "_SIZE = " << std::setbase(10) << sourceSize << ";" << std::endl;
		target << "const unsigned char " << variable << "[] = {" << std::endl;
	}

	target << std::setbase(16) << std::setfill('0');
	source.seekg(0, std::ios::beg);

	while (blockStart < sourceSize)
	{
		source.read(reinterpret_cast<char*>(buffer.get()), bufferSize);

		std::streamoff pos = source.tellg();
		std::streamoff blockSize = (pos < 0 ? sourceSize : pos) - blockStart;
		std::streamoff i = 0;

		while (i < blockSize)
		{
			size_t j = i;
			size_t end = std::min<size_t>(i + lineSize, blockSize);

			for (; j < end; j++)
				target << "0x" << std::setw(2) << (unsigned)buffer[j] << ",";

			target << std::endl;
			i = end;
		}

		blockStart += blockSize;
	}

	if (nullTerminate)
		target << "0x00,";

	if (variable.length() != 0)
	{
		target << "};" << std::endl;
	}
}

void PrintHelp()
{
	std::cout << "Usage: bin2hex {<option>}" << std::endl;
	std::cout << "Option format: -<option name>[ <option value>]" << std::endl;
	std::cout << std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << "[REQUIRED] -s[ource] {<path to file>}: files to be processed. Must be last argument." << std::endl;
	std::cout << "           -t[arget] <path>: target file. Default: std out." << std::endl;
	std::cout << "           -l[ine-size] <number>: bytes in one line. Default: 16." << std::endl;
	std::cout << "           -b[uffer] <number>: read buffer size. Default: 512." << std::endl;
	std::cout << "           -g[uard] <string>: guard name. Default: no guard" << std::endl;
}

void WrongUsage(const std::string& error)
{
	std::cout << "Wrong usage: " << error << std::endl;
	std::cout << std::endl;
	PrintHelp();
	exit(1);
}

void UnknownArgument(const std::string& arg)
{
	std::ostringstream ss;
	ss << "unknown argument - " << arg;
	WrongUsage(ss.str());
}

void Process(std::string sourcePath, std::ostream* target, size_t bufferSize, size_t lineSize)
{
	std::string fileName(sourcePath);
	std::string extension;
	auto extPos = fileName.rfind('.');
	auto slashPos = fileName.rfind('/');

	if (extPos != std::string::npos)
	{
		extension = fileName.substr(extPos + 1);
		fileName = fileName.substr(0, extPos);
	}

	if (slashPos != std::string::npos)
	{
		fileName = fileName.substr(slashPos + 1);
	}

	std::string variable(fileName);
	std::ifstream sourceFile(sourcePath, std::ios::in | std::ios::binary | std::ios::ate);

	if (!sourceFile.good())
	{
		std::cerr << "File not found: " << sourcePath << std::endl;
		exit(1);
	}

	std::transform(variable.begin(), variable.end(), variable.begin(), ::toupper);
	Bin2Hex(sourceFile, *target, variable, true, bufferSize, lineSize);
}

int main(int argsn, char** args)
{
	if (argsn == 1)
	{
		PrintHelp();
		return 2;
	}

	std::string sourcePath("./");
	std::string targetPath("");
	std::string guard("");
	size_t bufferSize = 512;
	size_t lineSize = 16;

	std::ofstream targetFile;
	std::ostream* target = &std::cout;

	size_t i = 0;
	while (++i < argsn && **args != '-')
	{
		std::string arg(args[i] + 1);
		std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);

		if (arg == "s" || arg == "source")
		{
			if (guard.length() > 0)
			{
				*target << "#ifndef " << guard << std::endl;
				*target << "#define " << guard << std::endl;
			}

			while (++i < argsn)
			{
				Process(args[i], target, bufferSize, lineSize);
			}

			if (guard.length() > 0)
			{
				*target << "#endif" << std::endl;
			}

			return 0;
		}
		else if (arg == "t" || arg == "target")
		{
			targetFile.open(args[++i], std::ios::out);
			target = &targetFile;
		}
		else if (arg == "l" || arg == "line-size")		lineSize = atol(args[++i]);
		else if (arg == "b" || arg == "buffer")			bufferSize = atol(args[++i]);
		else if (arg == "g" || arg == "guard")			guard = args[++i];
		else											UnknownArgument(arg);
	}

	WrongUsage("source key is required");
}

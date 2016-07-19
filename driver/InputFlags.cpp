#include <iomanip>
#include <vector>
#include <iostream>
#include "InputFlags.hpp"

/*
Copyright © 2014 Advanced Micro Devices, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following are met:

You must reproduce the above copyright notice.

Neither the name of the copyright holder nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific, prior, written permission from at least the copyright holder.

You must include the following terms in your license and/or other materials
provided with the software.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Without limiting the foregoing, the software may implement third party
technologies for which you must obtain licenses from parties other than AMD.
You agree that AMD has not obtained or conveyed to you, and that you shall be
responsible for obtaining the rights to use and/or distribute the applicable
underlying intellectual property rights related to the third party
technologies.  These third party technologies are not licensed hereunder.

If you use the software (in whole or in part), you shall adhere to all
applicable U.S., European, and other export laws, including but not limited to
the U.S. Export Administration Regulations (EAR) (15 C.F.R Sections
730-774), and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further,
pursuant to Section 740.6 of the EAR, you hereby certify that, except pursuant
to a license granted by the United States Department of Commerce Bureau of
Industry and Security or as otherwise permitted pursuant to a License Exception
under the U.S. Export Administration Regulations ("EAR"), you will not (1)
export, re-export or release to a national of a country in Country Groups D:1,
E:1 or E:2 any restricted technology, software, or source code you receive
hereunder, or (2) export to Country Groups D:1, E:1 or E:2 the direct product
of such technology or software, if such foreign produced direct product is
subject to national security controls as identified on the Commerce Control
List (currently found in Supplement 1 to Part 774 of EAR).  For the most
current Country Group listings, or for additional information about the EAR
or your obligations under those regulations, please refer to the U.S. Bureau
of Industry and Securitys website at http://www.bis.doc.gov/.
*/

InputFlags::InputFlags()
{
	AddInputFlag("help", 'h', "", "Print Help Message", "string");
}

void InputFlags::AddInputFlag(const std::string &_long_name,
							char _short_name,
							const std::string &_value,
							const std::string &_help_text,
							const std::string &_type)
{
	Input in;
	in.long_name = _long_name;
	in.short_name = _short_name;
	in.value = _value;
	in.help_text = _help_text;
	in.type = _type;

	if(MapInputs.count(_short_name) > 0)
		printf("Input flag: %s (%c) already exists !", _long_name.c_str(), _short_name);
	else
		MapInputs[_short_name] = in;
}

void InputFlags::Print()
{
	printf("SpMV Input Flags: \n\n");

	for(auto &content : MapInputs)
		std::cout<<std::setw(8)<<"--"<<content.second.long_name<<std::setw(20 - content.second.long_name.length())<<"-"<<content.first<<std::setw(8)<<" "<<content.second.help_text<<"\n";
	exit(0);
}

char InputFlags::FindShortName(const std::string &long_name)
{
	char short_name = '\0';

	for(auto &content : MapInputs)
	{
		if(content.second.long_name == long_name)
			short_name = content.first;
	}
	if(short_name == '\0')
	{
		std::cout<<"Long Name: "<<long_name<<" Not Found !";
		exit(0);
	}
	
	return short_name;
}

void InputFlags::Parse(int argc, char *argv[])
{
	std::vector<std::string> args;
	for(int i = 1; i < argc; i++)
		args.push_back(argv[i]);

	if(args.size() == 0) // No Input Flag
		Print();

	for(int i = 0; i < args.size(); i++)
	{
		std::string temp = args[i];
		if(temp[0] != '-')
		{
			printf("Illegal input flag\n");
			Print();
		}
		else if(temp[0] == '-' && temp[1] == '-') // Long Name Input
		{
			std::string long_name = temp.substr(2);
			if(long_name == "help")
				Print();

			char short_name = FindShortName(long_name);

			MapInputs[short_name].value = args[i+1];
			i++;
		}
		else if (temp[0] == '-' && temp[1] == '?') // Help Input
			Print();
		else // Short Name Input
		{
			char short_name = temp[1];
			if(MapInputs.find(short_name) == MapInputs.end())
			{
				std::cout<<"Input Flag: "<<short_name<<" Not Found !";
				exit(0);
			}
			if(short_name == 'h')
				Print();

			MapInputs[short_name].value = args[i+1];
			i++;
		}
	}
}

std::string InputFlags::GetValueStr(const std::string &long_name)
{
	char short_name = FindShortName(long_name);
	std::string value = MapInputs[short_name].value;

	return value;
}	

int InputFlags::GetValueInt(const std::string &long_name)
{
	char short_name = FindShortName(long_name);
	int value = atoi(MapInputs[short_name].value.c_str());

	return value;
}

uint64_t InputFlags::GetValueUint64(const std::string &long_name)
{
    char short_name = FindShortName(long_name);
    uint64_t value = strtoull(MapInputs[short_name].value.c_str(), NULL, 10);

    return value;
}

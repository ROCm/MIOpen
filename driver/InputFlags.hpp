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
 *******************************************************************************/#ifndef MIOPEN_INPUT_FLAGS_HPP_
#define MIOPEN_INPUT_FLAGS_HPP_

#include <string>
#include <map>

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

struct Input
{
	std::string long_name;
	char short_name;
	std::string value;
	std::string help_text;
	std::string type;
};

class InputFlags
{
	std::map<char, Input> MapInputs;

	public:
	InputFlags();
	void AddInputFlag(const std::string &_long_name, 
					char _short_name,
					const std::string &_value,
					const std::string &_help_text, 
					const std::string &type);
	void Parse(int argc, char *argv[]);
	char FindShortName(const std::string &_long_name) const;
	void Print() const;

	std::string GetValueStr(const std::string &_long_name) const;
	int GetValueInt(const std::string &_long_name) const;
	uint64_t GetValueUint64(const std::string &_long_name) const;
	double GetValueDouble(const std::string &_long_name) const;

	virtual ~InputFlags() {}
};

#endif //_MIOPEN_INPUT_FLAGS_HPP_

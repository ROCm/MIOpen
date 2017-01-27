#ifndef GUARD_MLOPEN_DEVICE_NAME_HPP
#define GUARD_MLOPEN_DEVICE_NAME_HPP

#include <string>
#include <map>

namespace mlopen {
	
std::string inline GetDeviceNameFromMap(std::string &name){

	static std::map<std::string, std::string> device_name_map = {
																	{"Fiji", "Fiji"},	
																	{"gfx802", "Fiji"},	
																	{"gfx803", "Fiji"},	
																	{"gfx804", "Fiji"},	
																};

	auto device_name_iterator = device_name_map.find(name);
	if(device_name_iterator != device_name_map.end()) {
		return device_name_iterator->second;
	}
	else {
		return name;
	}
}

} // namespace mlopen

#endif // GUARD_MLOPEN_DEVICE_NAME_HPP

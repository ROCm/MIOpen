#ifndef GUARD_MLOPEN_DEVICE_NAME_HPP
#define GUARD_MLOPEN_DEVICE_NAME_HPP

#include <string>
#include <map>

namespace mlopen {
    
std::string inline GetDeviceNameFromMap(std::string &name){

    static std::map<std::string, std::string> device_name_map = {
                                                                    {"Ellesmere", "gfx803"},
                                                                    {"Baffin", "gfx803"},
                                                                    {"RacerX", "gfx803"},
                                                                    {"Polaris10", "gfx803"},
                                                                    {"Polaris11", "gfx803"},
                                                                    {"Tonga", "gfx803"},
                                                                    {"Fiji", "gfx803"},   
                                                                    {"gfx800", "gfx803"}, 
                                                                    {"gfx802", "gfx803"}, 
                                                                    {"gfx803", "gfx803"}, 
                                                                    {"gfx804", "gfx803"}, 
                                                                    {"Vega10", "gfx900"}, 
                                                                    {"gfx900", "gfx900"}, 
                                                                    {"gfx901", "gfx900"}, 
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


#ifndef GUARD_MIOPEN_RETURNS_HPP
#define GUARD_MIOPEN_RETURNS_HPP

#define MIOPEN_RETURNS(...) -> decltype(__VA_ARGS__) { return __VA_ARGS__; }

#endif

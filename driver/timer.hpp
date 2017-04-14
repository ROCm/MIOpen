#ifndef GUARD_MIOPEN_TIMER_HPP
#define GUARD_MIOPEN_TIMER_HPP

#include <miopen/handle.hpp>
#include <chrono>

#define WALL_CLOCK inflags.GetValueInt("wall")

#define START_TIME \
	if(WALL_CLOCK) { \
	t.start(); }

#define STOP_TIME \
	if(WALL_CLOCK) {\
	t.stop(); }


class Timer
{
	public:
	Timer(){};
	void start() { st = std::chrono::steady_clock::now(); }
	void stop() { et = std::chrono::steady_clock::now(); }
	float gettime_ms() { return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(et-st).count(); }

	private:
	std::chrono::time_point<std::chrono::steady_clock> st;
	std::chrono::time_point<std::chrono::steady_clock> et;
};

#endif // GUARD_MIOPEN_TIMER_HPP

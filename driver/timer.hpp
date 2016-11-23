#ifndef GUARD_MLOPEN_TIMER_HPP
#define GUARD_MLOPEN_TIMER_HPP

#include <ctime>
#include <mlopen/handle.hpp>

#define WALL_CLOCK inflags.GetValueInt("wall")
#ifdef WIN32
#define CLOCK_MONOTONIC_RAW 0
static
int clock_gettime(int, struct timespec *tv)
{
	return timespec_get(tv, TIME_UTC);
}
#endif

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
	void start() { clock_gettime(CLOCK_MONOTONIC_RAW, &st); }
	void stop() { clock_gettime(CLOCK_MONOTONIC_RAW, &et); }
	float gettime_ms() { return ((et.tv_sec - st.tv_sec)*1e3 + (et.tv_nsec - st.tv_nsec)*1e-6); }

	private:
	struct timespec st;
	struct timespec et;
};

#endif // GUARD_MLOPEN_TIMER_HPP

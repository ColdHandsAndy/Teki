#ifndef TIME_MEASURER_HEADER
#define TIME_MEASURER_HEADER

#define CONC(a, b) a ## b

#define TIME_MEASURE_START(iterations, uniqueIndex) static int CONC(count, uniqueIndex) { 0 };\
													static std::array<std::chrono::microseconds, iterations> CONC(timings, uniqueIndex){};\
													auto CONC(start, uniqueIndex){ std::chrono::high_resolution_clock().now() };

#define TIME_MEASURE_END(iterations, uniqueIndex)  auto CONC(end, uniqueIndex){ std::chrono::high_resolution_clock().now() };\
							CONC(timings, uniqueIndex)[CONC(count, uniqueIndex)] = std::chrono::duration_cast<std::chrono::microseconds>(CONC(end, uniqueIndex) - CONC(start, uniqueIndex));\
							if (CONC(count, uniqueIndex)++ >= iterations - 1)\
							{\
								CONC(count, uniqueIndex) = 0;\
								std::chrono::microseconds CONC(average, uniqueIndex){};\
								for (auto timing : CONC(timings, uniqueIndex))\
								{\
									CONC(average, uniqueIndex) += timing;\
								}\
								std::cout << CONC(average, uniqueIndex) / iterations << " microseconds - average(100 iterations) function execution time." << std::endl;\
							}

#endif
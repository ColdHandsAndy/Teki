#ifndef ASSERTER_HEADER
#define ASSERTER_HEADER

#include <iostream>
#include <cassert>

#define EASSERT(cond, source, message) if(!(cond)) { std::cerr << '[' << source << ']' << " : " << message << std::endl; \
														 assert(false); }

#endif
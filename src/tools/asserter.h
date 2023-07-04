#ifndef ASSERTER_HEADER
#define ASSERTER_HEADER

#include <iostream>
#include <cassert>

#ifdef _DEBUG

#define ASSERT_DEBUG(cond, source, message) if(!(cond)) { std::cerr << '[' << source << ']' << " : " << message << std::endl; \
														 assert(false); }

#else

#define ASSERT_DEBUG(cond, source, message) (cond);

#endif

#define ASSERT_ALWAYS(cond, source, message) if(!(cond)) { std::cerr << '[' << source << ']' << " : " << message << std::endl; \
														 assert(false); }

#endif
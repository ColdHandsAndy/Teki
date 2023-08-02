#ifndef ARRAYSIZE_HEADER
#define ARRAYSIZE_HEADER

#define ARRAYSIZE(x) ((sizeof(x) / sizeof(*(x))) / static_cast<size_t>(!(sizeof(x) % sizeof(*(x)))))

#endif
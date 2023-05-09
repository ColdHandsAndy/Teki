#ifndef ALIGNMENT_HEADER
#define ALIGNMENT_HEADER

#define ALIGNED_SIZE(size, alignment_requirement) size + ((alignment_requirement - (size % alignment_requirement)) % alignment_requirement)

#endif
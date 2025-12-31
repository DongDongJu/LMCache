#include <cstdint>

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags);

void free_pinned_ptr(uintptr_t ptr);

uintptr_t alloc_pinned_numa_ptr(size_t size, int node);

void free_pinned_numa_ptr(uintptr_t ptr, size_t size);

// Unpinned (pageable) NUMA allocation: mmap + mbind + first-touch.
// This is intended for capacity tiers (e.g., CXL) where cudaHostRegister
// is undesirable.
uintptr_t alloc_numa_ptr(size_t size, int node);

void free_numa_ptr(uintptr_t ptr, size_t size);
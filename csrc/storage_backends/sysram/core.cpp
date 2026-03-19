// SPDX-License-Identifier: Apache-2.0

#include "core.h"

#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace {

void first_touch(void* ptr, size_t size) {
  const long page_size = sysconf(_SC_PAGESIZE);
  for (size_t offset = 0; offset < size; offset += static_cast<size_t>(page_size)) {
    volatile char* c = static_cast<volatile char*>(ptr) + offset;
    *c = 0;
  }
}

int mbind_sys(void* addr, unsigned long len, int mode,
              const unsigned long* nodemask, unsigned long maxnode,
              unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}

void* alloc_numa_buffer(size_t size, int numa_node) {
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error(std::string("mmap failed: ") + std::strerror(errno));
  }

  if (numa_node >= 0) {
    unsigned long mask = 1UL << numa_node;
    unsigned long maxnode = 8 * sizeof(mask);
    const int rc = mbind_sys(ptr, size, MPOL_BIND, &mask, maxnode,
                             MPOL_MF_MOVE | MPOL_MF_STRICT);
    if (rc != 0) {
      const int err = -rc;
      munmap(ptr, size);
      throw std::runtime_error(std::string("mbind failed: ") + std::strerror(err));
    }
  }

  first_touch(ptr, size);
  return ptr;
}

void free_numa_buffer(void* ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    return;
  }
  if (munmap(ptr, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + std::strerror(errno));
  }
}

}  // namespace

LMCacheSysRAMCore::LMCacheSysRAMCore(const std::vector<int>& numa_nodes,
                                     const std::vector<size_t>& pool_sizes_bytes,
                                     size_t slot_bytes)
    : slot_bytes_(slot_bytes) {
  if (numa_nodes.empty()) {
    throw std::invalid_argument("LMCacheSysRAMCore requires at least one NUMA pool");
  }
  if (numa_nodes.size() != pool_sizes_bytes.size()) {
    throw std::invalid_argument("numa_nodes and pool_sizes_bytes must have the same length");
  }
  if (slot_bytes_ == 0) {
    throw std::invalid_argument("slot_bytes must be greater than 0");
  }

  int64_t next_slot_id = 0;
  pools_.reserve(numa_nodes.size());
  pool_slot_bases_.reserve(numa_nodes.size());
  slot_in_use_.clear();

  for (size_t idx = 0; idx < numa_nodes.size(); ++idx) {
    const size_t raw_bytes = pool_sizes_bytes[idx];
    const size_t slot_count = raw_bytes / slot_bytes_;
    if (slot_count == 0) {
      throw std::invalid_argument("each SysRAM pool must fit at least one slot");
    }

    Pool pool;
    pool.numa_node = numa_nodes[idx];
    pool.size_bytes = raw_bytes;
    pool.slot_count = slot_count;
    pool.ptr = alloc_numa_buffer(raw_bytes, pool.numa_node);
    pool.buffer = torch::from_blob(
        pool.ptr,
        {static_cast<int64_t>(raw_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));

    pool.free_slots.reserve(slot_count);
    for (size_t slot = slot_count; slot-- > 0;) {
      pool.free_slots.push_back(slot);
      slot_locations_.push_back(SlotLocation{idx, slot});
      slot_in_use_.push_back(false);
    }

    pool_slot_bases_.push_back(next_slot_id);
    next_slot_id += static_cast<int64_t>(slot_count);
    total_capacity_bytes_ += slot_count * slot_bytes_;
    pools_.push_back(std::move(pool));
  }
}

LMCacheSysRAMCore::~LMCacheSysRAMCore() {
  for (auto& pool : pools_) {
    try {
      free_numa_buffer(pool.ptr, pool.size_bytes);
    } catch (...) {
      // Best-effort cleanup during destruction.
    }
    pool.ptr = nullptr;
  }
}

bool LMCacheSysRAMCore::slot_id_valid(int64_t slot_id) const {
  return slot_id >= 0 &&
         static_cast<size_t>(slot_id) < slot_locations_.size();
}

torch::Tensor LMCacheSysRAMCore::slot_tensor_unlocked(int64_t slot_id) const {
  const SlotLocation& loc = slot_locations_[static_cast<size_t>(slot_id)];
  const Pool& pool = pools_[loc.pool_index];
  const int64_t start =
      static_cast<int64_t>(loc.slot_index * slot_bytes_);
  return pool.buffer.narrow(0, start, static_cast<int64_t>(slot_bytes_));
}

std::optional<std::pair<int64_t, torch::Tensor>> LMCacheSysRAMCore::allocate_slot() {
  std::lock_guard<std::mutex> guard(mu_);
  for (size_t pool_idx = 0; pool_idx < pools_.size(); ++pool_idx) {
    Pool& pool = pools_[pool_idx];
    if (pool.free_slots.empty()) {
      continue;
    }

    const size_t slot_index = pool.free_slots.back();
    pool.free_slots.pop_back();
    const int64_t slot_id = pool_slot_bases_[pool_idx] +
                            static_cast<int64_t>(slot_index);
    slot_in_use_[static_cast<size_t>(slot_id)] = true;
    ++used_slots_;
    return std::make_pair(slot_id, slot_tensor_unlocked(slot_id));
  }
  return std::nullopt;
}

bool LMCacheSysRAMCore::release_slot(int64_t slot_id) {
  std::lock_guard<std::mutex> guard(mu_);
  if (!slot_id_valid(slot_id)) {
    return false;
  }
  if (!slot_in_use_[static_cast<size_t>(slot_id)]) {
    return false;
  }

  const SlotLocation& loc = slot_locations_[static_cast<size_t>(slot_id)];
  pools_[loc.pool_index].free_slots.push_back(loc.slot_index);
  slot_in_use_[static_cast<size_t>(slot_id)] = false;
  if (used_slots_ > 0) {
    --used_slots_;
  }
  return true;
}

bool LMCacheSysRAMCore::bind_key(const std::string& key, int64_t slot_id) {
  std::lock_guard<std::mutex> guard(mu_);
  if (!slot_id_valid(slot_id) || key_to_slot_.find(key) != key_to_slot_.end() ||
      !slot_in_use_[static_cast<size_t>(slot_id)]) {
    return false;
  }
  key_to_slot_[key] = slot_id;
  return true;
}

bool LMCacheSysRAMCore::contains(const std::string& key) const {
  std::lock_guard<std::mutex> guard(mu_);
  return key_to_slot_.find(key) != key_to_slot_.end();
}

bool LMCacheSysRAMCore::erase_key(const std::string& key) {
  std::lock_guard<std::mutex> guard(mu_);
  return key_to_slot_.erase(key) > 0;
}

std::optional<torch::Tensor> LMCacheSysRAMCore::get_tensor(
    const std::string& key) const {
  std::lock_guard<std::mutex> guard(mu_);
  auto it = key_to_slot_.find(key);
  if (it == key_to_slot_.end()) {
    return std::nullopt;
  }
  return slot_tensor_unlocked(it->second);
}

bool LMCacheSysRAMCore::copy_out(const std::string& key, torch::Tensor dst) const {
  TORCH_CHECK(dst.device().is_cpu(), "SysRAM copy_out destination must be on CPU");
  TORCH_CHECK(dst.is_contiguous(), "SysRAM copy_out destination must be contiguous");

  std::lock_guard<std::mutex> guard(mu_);
  auto it = key_to_slot_.find(key);
  if (it == key_to_slot_.end()) {
    return false;
  }

  torch::Tensor src = slot_tensor_unlocked(it->second);
  const size_t dst_bytes =
      static_cast<size_t>(dst.numel()) * static_cast<size_t>(dst.element_size());
  if (dst_bytes > slot_bytes_) {
    throw std::runtime_error("destination tensor is larger than the SysRAM slot");
  }

  std::memcpy(dst.data_ptr(), src.data_ptr(), dst_bytes);
  return true;
}

size_t LMCacheSysRAMCore::slot_bytes() const { return slot_bytes_; }

size_t LMCacheSysRAMCore::capacity_bytes() const { return total_capacity_bytes_; }

size_t LMCacheSysRAMCore::used_bytes() const {
  std::lock_guard<std::mutex> guard(mu_);
  return used_slots_ * slot_bytes_;
}

size_t LMCacheSysRAMCore::capacity_slots() const {
  return slot_locations_.size();
}

size_t LMCacheSysRAMCore::used_slots() const {
  std::lock_guard<std::mutex> guard(mu_);
  return used_slots_;
}

size_t LMCacheSysRAMCore::free_slots() const {
  std::lock_guard<std::mutex> guard(mu_);
  return slot_locations_.size() - used_slots_;
}

size_t LMCacheSysRAMCore::key_count() const {
  std::lock_guard<std::mutex> guard(mu_);
  return key_to_slot_.size();
}

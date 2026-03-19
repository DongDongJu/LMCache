// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/extension.h>

class LMCacheSysRAMCore {
 public:
  LMCacheSysRAMCore(const std::vector<int>& numa_nodes,
                    const std::vector<size_t>& pool_sizes_bytes,
                    size_t slot_bytes);
  ~LMCacheSysRAMCore();

  std::optional<std::pair<int64_t, torch::Tensor>> allocate_slot();
  bool release_slot(int64_t slot_id);
  bool bind_key(const std::string& key, int64_t slot_id);
  bool contains(const std::string& key) const;
  bool erase_key(const std::string& key);
  std::optional<torch::Tensor> get_tensor(const std::string& key) const;
  bool copy_out(const std::string& key, torch::Tensor dst) const;

  size_t slot_bytes() const;
  size_t capacity_bytes() const;
  size_t used_bytes() const;
  size_t capacity_slots() const;
  size_t used_slots() const;
  size_t free_slots() const;
  size_t key_count() const;

 private:
  struct Pool {
    int numa_node = -1;
    size_t size_bytes = 0;
    size_t slot_count = 0;
    void* ptr = nullptr;
    torch::Tensor buffer;
    std::vector<size_t> free_slots;
  };

  struct SlotLocation {
    size_t pool_index = 0;
    size_t slot_index = 0;
  };

  bool slot_id_valid(int64_t slot_id) const;
  torch::Tensor slot_tensor_unlocked(int64_t slot_id) const;

  mutable std::mutex mu_;
  size_t slot_bytes_ = 0;
  size_t total_capacity_bytes_ = 0;
  size_t used_slots_ = 0;
  std::vector<Pool> pools_;
  std::vector<SlotLocation> slot_locations_;
  std::vector<bool> slot_in_use_;
  std::vector<int64_t> pool_slot_bases_;
  std::unordered_map<std::string, int64_t> key_to_slot_;
};

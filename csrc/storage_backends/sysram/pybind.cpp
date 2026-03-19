// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_sysram, m) {
  m.doc() = "Native SysRAM backend core for LMCache";

  py::class_<LMCacheSysRAMCore>(m, "LMCacheSysRAMCore")
      .def(py::init<const std::vector<int>&, const std::vector<size_t>&, size_t>(),
           py::arg("numa_nodes"),
           py::arg("pool_sizes_bytes"),
           py::arg("slot_bytes"))
      .def("allocate_slot", &LMCacheSysRAMCore::allocate_slot,
           py::call_guard<py::gil_scoped_release>())
      .def("release_slot", &LMCacheSysRAMCore::release_slot,
           py::arg("slot_id"),
           py::call_guard<py::gil_scoped_release>())
      .def("bind_key", &LMCacheSysRAMCore::bind_key,
           py::arg("key"),
           py::arg("slot_id"))
      .def("contains", &LMCacheSysRAMCore::contains, py::arg("key"))
      .def("erase_key", &LMCacheSysRAMCore::erase_key, py::arg("key"))
      .def("get_tensor", &LMCacheSysRAMCore::get_tensor, py::arg("key"))
      .def("copy_out", &LMCacheSysRAMCore::copy_out,
           py::arg("key"),
           py::arg("dst"),
           py::call_guard<py::gil_scoped_release>())
      .def("slot_bytes", &LMCacheSysRAMCore::slot_bytes)
      .def("capacity_bytes", &LMCacheSysRAMCore::capacity_bytes)
      .def("used_bytes", &LMCacheSysRAMCore::used_bytes)
      .def("capacity_slots", &LMCacheSysRAMCore::capacity_slots)
      .def("used_slots", &LMCacheSysRAMCore::used_slots)
      .def("free_slots", &LMCacheSysRAMCore::free_slots)
      .def("key_count", &LMCacheSysRAMCore::key_count);
}

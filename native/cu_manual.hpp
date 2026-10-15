#pragma once
#include <pybind11/pybind11.h>

// Portable, in-memory reconstruction of CU's assisted line diameter geometry.
void register_cu_manual(pybind11::module_& module);

#pragma once
#include <pybind11/pybind11.h>
// Native geometry and graph loops; batch image primitives use installed cv2.
void register_cu_automatic(pybind11::module_& module);

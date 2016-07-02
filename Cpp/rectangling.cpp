#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_PLUGIN(rectangling) {
    py::module m("rectangling", "Colossus rectangling methods");

    return m.ptr();
}

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using Eigen::MatrixXi;

MatrixXi test_matrix_i(size_t n1, size_t n2)
{
    return MatrixXi::Constant(n1, n2, 42);
}


namespace py = pybind11;

PYBIND11_PLUGIN(rectangling) {
    py::module m("rectangling", "Colossus rectangling methods");

    m.def("test_matrix_i", &test_matrix_i, "create test matrix of 42s");

    return m.ptr();
}

#include <Eigen/Dense>
#include <trng/yarn2.hpp>
#include <trng/binomial_dist.hpp>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using Eigen::MatrixXi;
using Eigen::MatrixXd;
using rnd_engine_t = trng::yarn2;

MatrixXi test_matrix_i(size_t n1, size_t n2)
{
    return MatrixXi::Constant(n1, n2, 42);
}

MatrixXd test_matrix_d(size_t n1, size_t n2)
{
    return MatrixXd::Constant(n1, n2, 42.0);
}

MatrixXi excess_binomial_rnd(rnd_engine_t& rnd,
                             size_t n1, size_t n2,
                             double p, size_t n_obs_total)
{
    if (p <= 0.0 || p >= 1.0)
        throw std::range_error("p must be in (0, 1)");

    size_t n_entries = (n1 * n2);
    size_t n_obs_common = n_obs_total / n_entries;
    size_t n_obs_remaining = n_obs_total % n_entries;

    auto m = MatrixXi::Constant(n1, n2, -n_obs_common).eval();

    trng::binomial_dist binomial_bulk(p, n_obs_common);

    for (size_t i = 0; i != n1; ++i)
        for (size_t j = 0; j != n2; ++j)
            m(i, j) += 2 * binomial_bulk(rnd);

    trng::binomial_dist binomial_1(p, 1);

    for (size_t i = 0; n_obs_remaining; ++i)
        for (size_t j = 0; n_obs_remaining && (j != n2); ++j, --n_obs_remaining)
            m(i, j) += 2 * binomial_1(rnd) - 1;

    return m;
}


////////////////////////////////////////////////////////////////////////

class Observations
{
public:
    Observations(double zeta, const MatrixXi& theta)
        : zeta_(zeta), theta_(theta) {}

private:
    double zeta_;
    MatrixXi theta_;
};

class EngineContext
{
public:
    // The 'move' is to make sure the general 'generator&' template
    // doesn't match.  Gak.
    EngineContext(unsigned s) : rnd_(std::move(s)) {}

    size_t test_binomial(double p, size_t n_trials) {
        trng::binomial_dist distrn(p, n_trials);
        return distrn(rnd_);
    }

    MatrixXi excess_binomial_rnd(size_t n1, size_t n2,
                                 double p, size_t n_obs_total) {
        return ::excess_binomial_rnd(rnd_, n1, n2, p, n_obs_total);
    }

private:
    rnd_engine_t rnd_;
};

namespace py = pybind11;

PYBIND11_PLUGIN(rectangling) {
    py::module m("rectangling", "Colossus rectangling methods");

    m.def("test_matrix_i", &test_matrix_i, "create test matrix of 42s")
     .def("test_matrix_d", &test_matrix_d, "create test matrix of 42.0s");

    py::class_<EngineContext>(m, "EngineContext")
        .def(py::init<unsigned>())
        .def("test_binomial", &EngineContext::test_binomial)
        .def("excess_binomial_rnd", &EngineContext::excess_binomial_rnd)
        ;

    return m.ptr();
}

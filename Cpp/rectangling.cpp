#include <Eigen/Dense>
#include <trng/yarn2.hpp>
#include <trng/binomial_dist.hpp>
#include <stdexcept>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using Eigen::MatrixXi;
using Eigen::VectorXi;
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

    Observations(rnd_engine_t& rnd,
                 const VectorXi& chi1, const VectorXi& chi2,
                 double zeta, size_t n_observations);

    double zeta() const { return zeta_; }
    const MatrixXi& theta() const { return theta_; }

    double zeta_to_power(double x) const { return std::pow(zeta_, x); }

private:
    double zeta_;
    MatrixXi theta_;

    static MatrixXi random_theta_(rnd_engine_t& rnd,
                                  const VectorXi& chi1, const VectorXi& chi2,
                                  double zeta, size_t n_observations);
};

Observations::Observations(rnd_engine_t& rnd,
                           const VectorXi& chi1, const VectorXi& chi2,
                           double zeta, size_t n_observations)
    : Observations(zeta, random_theta_(rnd, chi1, chi2, zeta, n_observations))
{}

MatrixXi Observations::random_theta_(rnd_engine_t& rnd,
                                    const VectorXi& chi1, const VectorXi& chi2,
                                    double zeta, size_t n_observations)
{
    double delta = (zeta - 1.0) / (zeta + 1.0);
    double prob_dot = 0.5 * (1.0 + delta);
    auto de_chi_dots_over_crosses = excess_binomial_rnd(rnd, chi1.size(), chi2.size(),
                                                        prob_dot, n_observations);

    auto mk_mult = [](int x) { return (x ? -1 : 1); };
    auto chi1_mult = chi1.unaryExpr(mk_mult);
    auto chi2_mult = chi2.unaryExpr(mk_mult);
    auto chi_12 = chi1_mult * chi2_mult.transpose();

    return chi_12.array() * de_chi_dots_over_crosses.array();
}


////////////////////////////////////////////////////////////////////////

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

    Observations make_Observations(const VectorXi& chi1, const VectorXi& chi2,
                                   double zeta, size_t n_observations) {
        return Observations(rnd_, chi1, chi2, zeta, n_observations);
    }

private:
    rnd_engine_t rnd_;
};

namespace py = pybind11;

PYBIND11_PLUGIN(rectangling) {
    py::module m("rectangling", "Colossus rectangling methods");

    m.def("test_matrix_i", &test_matrix_i, "create test matrix of 42s")
     .def("test_matrix_d", &test_matrix_d, "create test matrix of 42.0s");

    py::class_<Observations>(m, "Observations")
        .def(py::init<double, const MatrixXi&>())
        .def_property_readonly("zeta", &Observations::zeta)
        .def_property_readonly("theta", &Observations::theta)
        .def("zeta_to_power", &Observations::zeta_to_power)
        ;

    py::class_<EngineContext>(m, "EngineContext")
        .def(py::init<unsigned>())
        .def("test_binomial", &EngineContext::test_binomial)
        .def("excess_binomial_rnd", &EngineContext::excess_binomial_rnd)
        .def("make_Observations", &EngineContext::make_Observations)
        ;

    return m.ptr();
}

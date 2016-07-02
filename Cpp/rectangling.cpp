#include <Eigen/Dense>
#include <trng/yarn2.hpp>
#include <trng/binomial_dist.hpp>
#include <trng/normal_dist.hpp>
#include <stdexcept>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using rnd_engine_t = trng::yarn2;

MatrixXi test_matrix_i(size_t n1, size_t n2)
{ return MatrixXi::Constant(n1, n2, 42); }

MatrixXd test_matrix_d(size_t n1, size_t n2)
{ return MatrixXd::Constant(n1, n2, 42.0); }

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

MatrixXd unit_normal_shaped_like(rnd_engine_t& rnd, const MatrixXi& theta)
{
    auto n1 = theta.rows();
    auto n2 = theta.cols();
    MatrixXd m(n1, n2);

    trng::normal_dist<double> normal(0.0, 1.0);

    for (auto i = 0; i != n1; ++i)
        for (auto j = 0; j != n2; ++j)
            m(i, j) = normal(rnd);

    return m;
}

VectorXi pattern_from_score(const VectorXd& scores)
{ return scores.unaryExpr([](double x) { return (x > 0.0) ? 1 : 0; }).cast<int>(); }

////////////////////////////////////////////////////////////////////////

class Observations
{
public:
    Observations(double zeta, const MatrixXi& theta);

    Observations(rnd_engine_t& rnd,
                 const VectorXi& chi1, const VectorXi& chi2,
                 double zeta, size_t n_observations);

    double zeta() const { return zeta_; }
    double log_zeta_recip() const { return log_zeta_recip_; }
    const MatrixXi& theta() const { return theta_; }
    const MatrixXd& z_pwr_theta() const { return z_pwr_theta_; }

    double zeta_to_power(double x) const { return std::pow(zeta_, x); }

    MatrixXd chk_f(const MatrixXd& x) const;

private:
    double zeta_;
    double log_zeta_recip_;
    MatrixXi theta_;
    MatrixXd z_pwr_theta_;

    static MatrixXi random_theta_(rnd_engine_t& rnd,
                                  const VectorXi& chi1, const VectorXi& chi2,
                                  double zeta, size_t n_observations);
};

Observations::Observations(double zeta, const MatrixXi& theta)
    : zeta_(zeta), log_zeta_recip_(1.0 / std::log(zeta)),
      theta_(theta), z_pwr_theta_(theta
                                  .cast<double>()
                                  .unaryExpr([this](double u) { return zeta_to_power(u); }))
{}

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

MatrixXd Observations::chk_f(const MatrixXd& x) const
{
    auto z_pwr_x = x.unaryExpr([this](double u) { return zeta_to_power(u); }).array();
    auto z_pwr_th = z_pwr_theta_.array();

    auto x_nrows = z_pwr_x.rows();
    auto x_ncols = z_pwr_x.cols();
    auto n_rows_match = (x_nrows == z_pwr_th.rows());
    auto n_cols_match = (x_ncols == z_pwr_th.cols());
    auto one_row = (x_nrows == 1);
    auto one_col = (x_ncols == 1);

    MatrixXd likelihood_ratio;
    if (n_rows_match && n_cols_match)
        likelihood_ratio = (1.0 + z_pwr_th * z_pwr_x) / (z_pwr_th + z_pwr_x);
    else if (n_rows_match && one_col) {
        auto z_pwr_th_cw = z_pwr_th.colwise();
        auto z_pwr_x_c0 = z_pwr_x.col(0);
        likelihood_ratio = (1.0 + z_pwr_th_cw * z_pwr_x_c0) / (z_pwr_th_cw + z_pwr_x_c0);
    } else if (n_cols_match && one_row) {
        auto z_pwr_th_rw = z_pwr_th.rowwise();
        auto z_pwr_x_r0 = z_pwr_x.row(0);
        likelihood_ratio = (1.0 + z_pwr_th_rw * z_pwr_x_r0) / (z_pwr_th_rw + z_pwr_x_r0);
    } else
        throw std::invalid_argument("incompatible dimensions of 'x' wrt 'theta'");

    auto log_likelihood_ratio = likelihood_ratio.unaryExpr([](double u) { return std::log(u); });
    return (log_zeta_recip_ * log_likelihood_ratio);
}

////////////////////////////////////////////////////////////////////////

class FactorGraphState
{
public:
    FactorGraphState(const Observations& obs,
                     const MatrixXd& score_1,
                     const MatrixXd& score_2);

    FactorGraphState(rnd_engine_t& rnd, const Observations& obs);

    void update_score_1();
    void update_score_2();

    const MatrixXd& score_1() const { return score_1_; }
    const MatrixXd& score_2() const { return score_2_; }

    VectorXd s1() const;
    VectorXd s2() const;

    VectorXi pattern_1() const { return pattern_from_score(s1()); }
    VectorXi pattern_2() const { return pattern_from_score(s2()); }

private:
    const Observations& obs_;

    // score_1[i, j] is message sent from K1i to check node linking K1i and K2j (and K12ij).
    // score_2[i, j] is message sent from K2j to check node linking K2j and K1i (and K12ij).
    //
    MatrixXd score_1_, score_2_;
};

FactorGraphState::FactorGraphState(const Observations& obs,
                                   const MatrixXd& score_1,
                                   const MatrixXd& score_2)
    : obs_(obs), score_1_(score_1), score_2_(score_2)
{}

FactorGraphState::FactorGraphState(rnd_engine_t& rnd, const Observations& obs)
    : FactorGraphState(obs,
                       unit_normal_shaped_like(rnd, obs.theta()),
                       unit_normal_shaped_like(rnd, obs.theta()))
{}

void FactorGraphState::update_score_1()
{
    auto msgs = obs_.chk_f(score_2_);
    auto msg_sums = msgs.rowwise().sum();
    score_1_ = (-msgs).colwise() + msg_sums;
}

void FactorGraphState::update_score_2()
{
    auto msgs = obs_.chk_f(score_1_);
    auto msg_sums = msgs.colwise().sum();
    score_2_ = (-msgs).rowwise() + msg_sums;
}

VectorXd FactorGraphState::s1() const
{
    auto msgs = obs_.chk_f(score_2_);
    return msgs.rowwise().sum();
}

VectorXd FactorGraphState::s2() const
{
    auto msgs = obs_.chk_f(score_1_);
    return msgs.colwise().sum();
}

////////////////////////////////////////////////////////////////////////

class AccurateConvergenceState
{
private:
    const Observations& obs_;
    VectorXd score_1_;
    RowVectorXd score_2_;
};

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
                                 double p, size_t n_obs_total)
    { return ::excess_binomial_rnd(rnd_, n1, n2, p, n_obs_total); }

    MatrixXd unit_normal_shaped_like(const MatrixXi& theta)
    { return ::unit_normal_shaped_like(rnd_, theta); }

    Observations make_Observations(const VectorXi& chi1, const VectorXi& chi2,
                                   double zeta, size_t n_observations)
    { return Observations(rnd_, chi1, chi2, zeta, n_observations); }

    FactorGraphState make_FactorGraphState(const Observations& obs)
    { return FactorGraphState(rnd_, obs); }

private:
    rnd_engine_t rnd_;
};

namespace py = pybind11;

PYBIND11_PLUGIN(rectangling) {
    py::module m("rectangling", "Colossus rectangling methods");

    m // Formatting here a bit odd but makes 'def()'s line up
        .def("test_matrix_i", &test_matrix_i, "create test matrix of 42s")
        .def("test_matrix_d", &test_matrix_d, "create test matrix of 42.0s")
        ;

    py::class_<Observations>(m, "Observations")
        .def(py::init<double, const MatrixXi&>())
        .def_property_readonly("zeta", &Observations::zeta)
        .def_property_readonly("log_zeta_recip", &Observations::log_zeta_recip)
        .def_property_readonly("z_pwr_theta", &Observations::z_pwr_theta)
        .def_property_readonly("theta", &Observations::theta)
        .def("zeta_to_power", &Observations::zeta_to_power)
        .def("chk_f", &Observations::chk_f)
        ;

    py::class_<FactorGraphState>(m, "FactorGraphState")
        .def(py::init<const Observations&, const MatrixXd&, const MatrixXd&>())
        .def_property_readonly("score_1", &FactorGraphState::score_1)
        .def_property_readonly("score_2", &FactorGraphState::score_2)
        .def("update_score_1", &FactorGraphState::update_score_1)
        .def("update_score_2", &FactorGraphState::update_score_2)
        .def_property_readonly("s1", &FactorGraphState::s1)
        .def_property_readonly("s2", &FactorGraphState::s2)
        .def_property_readonly("pattern_1", &FactorGraphState::pattern_1)
        .def_property_readonly("pattern_2", &FactorGraphState::pattern_2)
        ;

    py::class_<AccurateConvergenceState>(m, "AccurateConvergenceState")
        ;

    py::class_<EngineContext>(m, "EngineContext")
        .def(py::init<unsigned>())
        .def("test_binomial", &EngineContext::test_binomial)
        .def("excess_binomial_rnd", &EngineContext::excess_binomial_rnd)
        .def("unit_normal_shaped_like", &EngineContext::unit_normal_shaped_like)
        .def("make_Observations", &EngineContext::make_Observations)
        .def("make_FactorGraphState", &EngineContext::make_FactorGraphState)
        ;

    return m.ptr();
}

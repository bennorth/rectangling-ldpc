#include <Eigen/Dense>
#include <trng/yarn2.hpp>
#include <trng/binomial_dist.hpp>
#include <trng/normal_dist.hpp>
#include <stdexcept>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using VectorXu = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;

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

    size_t i1 = 0, i2 = 0;
    while (n_obs_remaining--) {
        m(i1, i2) += 2 * binomial_1(rnd) - 1;
        i1 = (i1 + 1) % n1;
        i2 = (i2 + 1) % n2;
    }

    return m;
}

MatrixXd unit_normal_of_size(rnd_engine_t& rnd, size_t n1, size_t n2)
{
    MatrixXd m(n1, n2);

    trng::normal_dist<double> normal(0.0, 1.0);

    for (size_t i = 0; i != n1; ++i)
        for (size_t j = 0; j != n2; ++j)
            m(i, j) = normal(rnd);

    return m;
}

MatrixXd unit_normal_shaped_like(rnd_engine_t& rnd, const MatrixXi& theta)
{ return unit_normal_of_size(rnd, theta.rows(), theta.cols()); }

VectorXi pattern_from_score(const VectorXd& scores)
{ return scores.unaryExpr([](double x) { return (x > 0.0) ? 1 : 0; }).cast<int>(); }

////////////////////////////////////////////////////////////////////////

class Patterns
{
public:
    Patterns(const VectorXi& chi1, const VectorXi& chi2) : chi1(chi1), chi2(chi2) {}
    const VectorXi chi1, chi2;
};

////////////////////////////////////////////////////////////////////////

class Observations
{
public:
    Observations(double zeta, const MatrixXi& theta);

    Observations(rnd_engine_t& rnd,
                 const VectorXi& chi1, const VectorXi& chi2,
                 double zeta, size_t n_observations);

    Observations(rnd_engine_t& rnd, const Patterns& patterns,
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

Observations::Observations(rnd_engine_t& rnd,
                           const Patterns& patterns,
                           double zeta, size_t n_observations)
    : Observations(rnd, patterns.chi1, patterns.chi2, zeta, n_observations)
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
public:
    AccurateConvergenceState(const Observations& obs,
                             const VectorXd& score_1,
                             const RowVectorXd& score_2);

    AccurateConvergenceState(rnd_engine_t& rnd, const Observations& obs);

    void update_score_1();
    void update_score_2();

    const VectorXd& score_1() const { return score_1_; }
    const VectorXd& s1() const { return score_1_; }
    const RowVectorXd& score_2() const { return score_2_; }
    const RowVectorXd& s2() const { return score_2_; }

    VectorXi pattern_1() const { return pattern_from_score(score_1_); }
    VectorXi pattern_2() const { return pattern_from_score(score_2_.transpose()); }

private:
    const Observations& obs_;
    VectorXd score_1_;
    RowVectorXd score_2_;

    static const Observations& verify_dimensions_consistent(const Observations& obs,
                                                            const VectorXd& score_1,
                                                            const RowVectorXd& score_2);
};

AccurateConvergenceState::AccurateConvergenceState(const Observations& obs,
                                                   const VectorXd& score_1,
                                                   const RowVectorXd& score_2)
    : obs_(verify_dimensions_consistent(obs, score_1, score_2)),
      score_1_(score_1),
      score_2_(score_2)
{}

AccurateConvergenceState::AccurateConvergenceState(rnd_engine_t& rnd, const Observations& obs)
    : AccurateConvergenceState(obs,
                               unit_normal_of_size(rnd, obs.theta().rows(), 1),
                               unit_normal_of_size(rnd, 1, obs.theta().cols()))
{}

const Observations& AccurateConvergenceState::verify_dimensions_consistent(
    const Observations& obs, const VectorXd& score_1, const RowVectorXd& score_2)
{
    if ((obs.theta().rows() != score_1.size()) || (obs.theta().cols() != score_2.size()))
        throw std::invalid_argument("incompatible dimensions of obs and scores");

    return obs;
}

void AccurateConvergenceState::update_score_1()
{
    auto msgs = obs_.chk_f(score_2_);
    score_1_ = msgs.rowwise().sum();
}

void AccurateConvergenceState::update_score_2()
{
    auto msgs = obs_.chk_f(score_1_);
    score_2_ = msgs.colwise().sum();
}

////////////////////////////////////////////////////////////////////////

template<typename State>
bool update_until_convergence(State& state, size_t n_same_converged, size_t max_n_iter)
{
    VectorXi prev_patterns_1 = state.pattern_1();
    VectorXi prev_patterns_2 = state.pattern_2();

    size_t n_same = 1;
    size_t n_iterations = 0;

    while (n_iterations < max_n_iter && n_same < n_same_converged) {
        state.update_score_1();
        state.update_score_2();

        VectorXi patterns_1 = state.pattern_1();
        VectorXi patterns_2 = state.pattern_2();

        if (patterns_1 == prev_patterns_1 && patterns_2 == prev_patterns_2) {
            n_same += 1;
        } else {
            prev_patterns_1 = patterns_1;
            prev_patterns_2 = patterns_2;
            n_same = 1;
        }
        n_iterations += 1;
    }

    return (n_same == n_same_converged);
}

////////////////////////////////////////////////////////////////////////

/*
  Generating legal wheel patterns for Chi1 (41 cams) and Chi2 (31 cams)

  D-Chi1 must have as near to 20.5 crosses as possible, while having an
  even number of crosses.  So it must have 20 crosses.

  D-Chi2 similarly must have 16 crosses.

  The un-delta'd wheels then must have:

  Chi1: 20 crosses and 21 dots
     or 21 crosses and 20 dots;

  Chi2: 15 crosses and 16 dots
     or 16 crosses and 15 dots.

  Cannot have more than four consecutive like characters in the
  un-delta'd wheel, i.e., no more than three consecutive dots in the
  delta'd-wheel.

  [25D(e), p.148]

  Adapting the analysis of [25X, p.183] to Chi1:

  Chi1 must consist of 10 blocks of cross and 10 blocks of dot.  Each
  consists of from 1 to 4 characters.  Either the sum of the cross-block
  sizes must be 20 and the sum of the dot-block sizes 21, or the other
  way round.  Therefore we choose ten numbers from {1, 2, 3, 4} which
  sum to 20, and ten which sum to 21, and interleave them.  Then choose
  whether we start with a dot- or cross-block.  Then choose a random
  starting position.  (For sampling, we can ignore the redundancy by a
  factor of 10 arising from choosing which block to start with.)
 */

/*
  Discrete Dirichlet distribution

  Paper has everything in terms of positive integers, but I think the
  implementation become simpler if we think of it as non-negative,
  adjusting everything afterwards.  I.e., an original 'required sum' of
  20 from 8 numbers in {1, 2, 3, 4} because a required sum of 20-8 = 12
  from 8 numbers in {0, 1, 2, 3}.  Then the user can add one to each
  term afterwards.
*/

class DirichletState
{
public:
    enum class Bound { Lower, Upper };
    DirichletState(size_t n_terms, size_t max_term, size_t required_sum, Bound bound);
    DirichletState(const VectorXu& terms, size_t max_term);

    const VectorXu& terms() const { return terms_; }
    void mutate(double x);
    void mutate_many(const std::vector<double>& xs);

private:
    const size_t n_terms_;
    const size_t max_term_;
    const size_t required_sum_;
    VectorXu terms_;
};

DirichletState::DirichletState(size_t n_terms, size_t max_term, size_t required_sum, Bound bound)
    : n_terms_(n_terms), max_term_(max_term), required_sum_(required_sum)
{
    if (n_terms * max_term < required_sum)
        throw std::range_error("inconsistent arguments: required_sum too large");

    size_t n_maximal = required_sum / max_term;
    size_t left_over = required_sum % max_term;

    terms_.resize(n_terms);
    for (size_t i = 0; i < n_maximal; ++i)
        terms_(i) = max_term;

    // If it exactly fits (e.g., 12 = 4 + 4 + 4), there are no more
    // terms to populate, and also Upper and Lower are equal, so we're
    // finished.
    if (n_maximal == n_terms)
        return;

    for (size_t i = n_maximal + 1; i < n_terms; ++i)
        terms_(i) = 0;

    terms_(n_maximal) = left_over;

    if (bound == Bound::Lower)
        terms_.reverseInPlace();
}

DirichletState::DirichletState(const VectorXu& terms, size_t max_term)
    : n_terms_(terms.rows()), max_term_(max_term), required_sum_(terms.sum()), terms_(terms)
{
}

void DirichletState::mutate(double x)
{
    if ((x < 0.0) || (x >= static_cast<double>(n_terms_ - 1)))
        throw std::range_error("x out of range [0, n-1)");

    double integer_part;
    double fractional_part = std::modf(x, &integer_part);

    size_t first_elt_idx = static_cast<size_t>(integer_part);
    size_t pair_sum = terms_(first_elt_idx) + terms_(first_elt_idx + 1);

    size_t min_choice_0 = (pair_sum < max_term_) ? 0 : (pair_sum - max_term_);
    size_t max_choice_0 = std::min(pair_sum, max_term_);
    size_t n_choices = max_choice_0 - min_choice_0 + 1;

    size_t choice_offset = static_cast<size_t>(n_choices * fractional_part);
    size_t choice_0 = min_choice_0 + choice_offset;

    terms_(first_elt_idx) = choice_0;
    terms_(first_elt_idx + 1) = pair_sum - choice_0;
}

void DirichletState::mutate_many(const std::vector<double>& xs)
{
    size_t n_elts = xs.size();
    while (n_elts)
        mutate(xs[--n_elts]);
}

////////////////////////////////////////////////////////////////////////

class DirichletSamplingState
{
public:
    DirichletSamplingState(size_t n_terms, size_t max_term, size_t required_sum);
    VectorXu maybe_coalesced_result() const;

private:
    const size_t n_terms_;
    const size_t max_term_;
    const size_t required_sum_;
    std::vector<double> vec_lambda_;
    VectorXu maybe_coalesced_result_;
    bool has_coalesced_;

    void determine_coalescence_();
};

DirichletSamplingState::DirichletSamplingState(size_t n_terms, size_t max_term, size_t required_sum)
    : n_terms_(n_terms), max_term_(max_term), required_sum_(required_sum)
{
    determine_coalescence_();
}

void DirichletSamplingState::determine_coalescence_()
{
    DirichletState x_lwr(n_terms_, max_term_, required_sum_,
                         DirichletState::Bound::Lower);
    DirichletState x_upr(n_terms_, max_term_, required_sum_,
                         DirichletState::Bound::Upper);

    x_lwr.mutate_many(vec_lambda_);
    x_upr.mutate_many(vec_lambda_);

    // Set this anyway; it will be ignored if they're not the same.
    maybe_coalesced_result_ = x_lwr.terms();
    has_coalesced_ = (x_lwr.terms() == x_upr.terms());
}

VectorXu DirichletSamplingState::maybe_coalesced_result() const
{
    if (has_coalesced_)
        return maybe_coalesced_result_;
    else
        return VectorXu();
}

////////////////////////////////////////////////////////////////////////

class DirichletSamplingRun
{
public:
    DirichletSamplingRun(rnd_engine_t& rnd,
                         size_t n_terms, size_t max_term, size_t required_sum);

private:
    rnd_engine_t& rnd_;
    DirichletSamplingState state_;
};

DirichletSamplingRun::DirichletSamplingRun(rnd_engine_t& rnd,
                                           size_t n_terms, size_t max_term, size_t required_sum)
    : rnd_(rnd), state_(n_terms, max_term, required_sum)
{
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
                                 double p, size_t n_obs_total)
    { return ::excess_binomial_rnd(rnd_, n1, n2, p, n_obs_total); }

    MatrixXd unit_normal_shaped_like(const MatrixXi& theta)
    { return ::unit_normal_shaped_like(rnd_, theta); }

    MatrixXd unit_normal_of_size(size_t n1, size_t n2)
    { return ::unit_normal_of_size(rnd_, n1, n2); }

    Observations make_Observations_chis(const VectorXi& chi1, const VectorXi& chi2,
                                        double zeta, size_t n_observations)
    { return Observations(rnd_, chi1, chi2, zeta, n_observations); }

    Observations make_Observations_pat(const Patterns& patterns,
                                       double zeta, size_t n_observations)
    { return Observations(rnd_, patterns, zeta, n_observations); }

    FactorGraphState make_FactorGraphState(const Observations& obs)
    { return FactorGraphState(rnd_, obs); }

    AccurateConvergenceState make_AccurateConvergenceState(const Observations& obs)
    { return AccurateConvergenceState(rnd_, obs); }

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

    py::class_<Patterns>(m, "Patterns")
        .def(py::init<const VectorXi&, const VectorXi&>())
        .def_readonly("chi1", &Patterns::chi1)
        .def_readonly("chi2", &Patterns::chi2)
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
        .def(py::init<const Observations&, const VectorXd&, const VectorXd&>())
        .def_property_readonly("s1", &AccurateConvergenceState::s1)
        .def_property_readonly("score_1", &AccurateConvergenceState::score_1)
        .def_property_readonly("s2", &AccurateConvergenceState::s2)
        .def_property_readonly("score_2", &AccurateConvergenceState::score_2)
        .def_property_readonly("pattern_1", &AccurateConvergenceState::pattern_1)
        .def_property_readonly("pattern_2", &AccurateConvergenceState::pattern_2)
        .def("update_score_1", &AccurateConvergenceState::update_score_1)
        .def("update_score_2", &AccurateConvergenceState::update_score_2)
        ;

    py::class_<DirichletState> cls_DirichletState(m, "DirichletState");
    cls_DirichletState
        .def(py::init<const VectorXu&, size_t>())
        .def(py::init<size_t, size_t, size_t, DirichletState::Bound>())
        .def_property_readonly("terms", &DirichletState::terms)
        .def("mutate", &DirichletState::mutate)
        .def("mutate_many", &DirichletState::mutate_many)
        ;

    py::enum_<DirichletState::Bound>(cls_DirichletState, "Bound")
        .value("Lower", DirichletState::Bound::Lower)
        .value("Upper", DirichletState::Bound::Upper)
        ;

    py::class_<DirichletSamplingState>(m, "DirichletSamplingState")
        .def(py::init<size_t, size_t, size_t>())
        .def("maybe_coalesced_result", &DirichletSamplingState::maybe_coalesced_result)
        ;

    py::class_<DirichletSamplingRun>(m, "DirichletSamplingRun")
        ;

    py::class_<EngineContext>(m, "EngineContext")
        .def(py::init<unsigned>())
        .def("test_binomial", &EngineContext::test_binomial)
        .def("excess_binomial_rnd", &EngineContext::excess_binomial_rnd)
        .def("unit_normal_shaped_like", &EngineContext::unit_normal_shaped_like)
        .def("unit_normal_of_size", &EngineContext::unit_normal_of_size)
        .def("make_Observations", &EngineContext::make_Observations_chis)
        .def("make_Observations", &EngineContext::make_Observations_pat)
        .def("make_FactorGraphState", &EngineContext::make_FactorGraphState)
        .def("make_AccurateConvergenceState", &EngineContext::make_AccurateConvergenceState)
        ;

    m // Formatting....
        .def("converge_FGS", &update_until_convergence<FactorGraphState>)
        .def("converge_ACS", &update_until_convergence<AccurateConvergenceState>);

    return m.ptr();
}

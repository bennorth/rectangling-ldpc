import numpy as np
import numpy.testing as nptest
import re
from collections import Counter
from toolz.functoolz import compose
from operator import itemgetter as nth
from itertools import groupby
import rectangling as cr
import py_rectangling as pr
import pytest

def test_import():
    """
    If we get this far, it was successful.
    """
    assert True

@pytest.mark.parametrize(
    'function_name, expected_kind',
    [('test_matrix_i', 'i'), ('test_matrix_d', 'f')],
    ids=['Xi', 'Xd'])
#
def test_matrix_interchange(function_name, expected_kind):
    shape = (7, 13)
    fn = getattr(cr, function_name)
    mx = fn(*shape)
    assert mx.shape == shape
    assert mx.dtype.kind == expected_kind
    nptest.assert_array_equal(mx, 42)


@pytest.mark.parametrize(
    'n, exp_rot_xs',
    [(0, [0, 1, 2, 3, 4]),
     (1, [1, 2, 3, 4, 0]),
     (2, [2, 3, 4, 0, 1]),
     (3, [3, 4, 0, 1, 2]),
     (4, [4, 0, 1, 2, 3]),
     (5, [0, 1, 2, 3, 4]),
     (8, [3, 4, 0, 1, 2]),
     ],
    ids=compose(str, nth(0)))
#
def test_vector_rotate(n, exp_rot_xs):
    nptest.assert_array_equal(cr.rotate(np.arange(5), n),
                              np.array(exp_rot_xs))


def test_vector_delta_wheel():
    ns = np.arange(100)
    # Pseudo-random strings of 0s and 1s
    xs = (((ns % 5) > 3) ^ ((ns % 7) > 4)).astype(int)
    nptest.assert_array_equal(cr.delta_wheel(xs), TestPatterns.delta_wheel(xs))


def test_vector_delta_wheel_trivial_cases():
    assert cr.delta_wheel(np.array([], dtype=int)).size == 0
    single_0 = np.array([0], dtype=int)
    nptest.assert_array_equal(cr.delta_wheel(np.array([0], dtype=int)), single_0)
    nptest.assert_array_equal(cr.delta_wheel(np.array([1], dtype=int)), single_0)


@pytest.fixture
def engine_context():
    return cr.EngineContext(42)


def py_Observations(obs):
    return pr.Observations(obs.zeta, obs.theta)


class TestEngineContext:
    def test_construction(self):
        cr.EngineContext(42)

    def test_binomial(self, engine_context):
        n_successes = engine_context.test_binomial(0.8, 10000)
        # Value chosen to make test pass, but sanity-checking that it's
        # near 8000.
        assert n_successes == 8029

    def test_uniform(self, engine_context):
        xs = engine_context.uniform_of_size(1000, 20.0)
        # Values chosen to make test pass, but sanity-checking that
        # they're near expected.
        assert np.sum(xs < 5.0) == 243  # exp. 250
        assert np.sum(xs < 10.0) == 494  # exp. 500
        assert np.sum(xs < 15.0) == 730  # exp. 750

    def test_excess_binomial_rnd_bad_input(self, engine_context):
        pytest.raises_regexp(ValueError, r'must be in \(0, 1\)',
                             engine_context.excess_binomial_rnd, 2, 3, -0.5, 10)

    @pytest.mark.parametrize('p', [0.00001, 0.99999], ids='01')
    @pytest.mark.parametrize('n_leftover', range(6))
    def test_excess_binomial_rnd(self, engine_context, n_leftover, p):
        n_excess = engine_context.excess_binomial_rnd(2, 3, p, 600 + n_leftover)
        raw_exp_n_excess = np.full((2, 3), 100, dtype='i')
        unwrapped_idxs = np.arange(n_leftover, dtype=np.int32)
        raw_exp_n_excess[unwrapped_idxs % 2, unwrapped_idxs % 3] += 1
        exp_n_excess = (raw_exp_n_excess if p > 0.5 else -raw_exp_n_excess)
        nptest.assert_array_equal(n_excess, exp_n_excess)

    def test_normal_like(self, engine_context):
        dummy_theta = np.zeros((3, 5), dtype='i')
        zs = engine_context.unit_normal_shaped_like(dummy_theta)
        assert zs.shape == dummy_theta.shape

    def test_normal_of_size(self, engine_context):
        zs = engine_context.unit_normal_of_size(3, 4)
        assert zs.shape == (3, 4)

@pytest.fixture
def sample_chi1():
    return np.array([0, 1, 0], dtype='i')

@pytest.fixture
def sample_chi2():
    return np.array([0, 1, 1, 0], dtype='i')

@pytest.fixture
def zero_chi1():
    return np.array([0, 0, 0], dtype='i')

@pytest.fixture
def zero_chi2():
    return np.array([0, 0, 0, 0], dtype='i')

@pytest.fixture
def zero_obs():
    return cr.Observations(1.25, np.zeros((3, 4), dtype='i'))

@pytest.fixture
def sample_obs(engine_context, sample_chi1, sample_chi2):
    return engine_context.make_Observations(sample_chi1, sample_chi2, 1.25, 9000)


class TestWheelPattern:
    def test_construction(self):
        xs = np.array([1, 0, 1, 1, 0])
        p = cr.WheelPattern(xs)
        nptest.assert_array_equal(p.xs, xs)
        assert len(p) == len(xs)

    def test_max_consecutive_same(self):
        assert isinstance(cr.WheelPattern.max_consecutive_same, int)

    @pytest.mark.parametrize(
        'xs, exp_mrl, exp_mdrl',
        [([], 0, 0),
         ([1], 1, 0),
         ([0], 1, 1),
         ([1, 1, 1, 0, 0, 1, 0], 3, 2),
         ([0, 1, 1, 1, 0, 0, 1, 0, 0], 3, 3),
         ([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0], 3, 2),
         ([1] * 5, 5, 0),
         ([0] * 5, 5, 5),
         ([1] * 500, 500, 0),
         ([0] * 500, 500, 500),
         ([1, 1, 0, 0, 0, 1, 1], 4, 3),
         ([0, 0, 1, 1, 1, 0, 0], 4, 4)],
        ids=['empty', 'single-1', 'single-0',
             'arbitrary-a', 'arbitrary-b', 'arbitrary-c',
             'five-1', 'five-0', 'lots-1', 'lots-0',
             'wrapped-1', 'wrapped-0'])
    def test_max_run_length(self, xs, exp_mrl, exp_mdrl):
        xs_arr = np.array(xs, dtype=int)
        assert cr.WheelPattern.max_run_length(xs, 2) == exp_mrl
        assert cr.WheelPattern.max_run_length(xs, 1) == exp_mdrl

    # Test cases taken from 25D(e).  The two 'expected number of crosses in delta-chi'
    # correspond to a 'low' and 'high' value of the uniform variate respectively.  Often
    # these values are the same.  The case 'n % 4 == 0' doesn't occur amongst the real
    # chi wheels, but we make up a test case for it anyway.
    @pytest.mark.parametrize(
        'n, exp_ns_cross',
        [(41, [20, 20]),  # n % 4 == 1
         (31, [16, 16]),  # n % 4 == 3
         (29, [14, 14]),  # n % 4 == 1
         (26, [12, 14]),  # n % 4 == 2
         (23, [12, 12]),  # n % 4 == 3
         (24, [12, 12]),  # n % 4 == 0
         ],
        ids=compose(str, nth(0)))
    #
    def test_n_cross_in_delta(self, n, exp_ns_cross):
        assert [n % 2 for n in exp_ns_cross] == [0, 0]
        assert cr.WheelPattern.n_cross_in_delta(n, 0.25) == exp_ns_cross[0]
        assert cr.WheelPattern.n_cross_in_delta(n, 0.75) == exp_ns_cross[1]

    @pytest.mark.parametrize(
        'n, exp_ns_cross',
        [(41, [20, 21]),  # n % 4 == 1
         (31, [15, 16]),  # n % 4 == 3
         (29, [14, 15]),  # n % 4 == 1
         (26, [13, 13]),  # n % 4 == 2
         (23, [11, 12]),  # n % 4 == 3
         (24, [12, 12]),  # n % 4 == 0
         ],
        ids=compose(str, nth(0)))
    #
    def test_n_cross_in_un_delta(self, n, exp_ns_cross):
        assert cr.WheelPattern.n_cross_in_un_delta(n, 0.25) == exp_ns_cross[0]
        assert cr.WheelPattern.n_cross_in_un_delta(n, 0.75) == exp_ns_cross[1]

    def test_interleave_crosses_dots(self):
        ns_cross = np.array([1, 4, 3, 1, 2])
        ns_dot = np.array([3, 2, 2, 1, 2])
        pattern = cr.WheelPattern.interleave_crosses_dots(ns_cross, ns_dot)

        # Should start with a 'cross':
        assert pattern[0] == 1

        run_lengths = np.array([len(list(g)) for x, g in groupby(pattern)])
        nptest.assert_array_equal(run_lengths[::2], ns_cross)
        nptest.assert_array_equal(run_lengths[1::2], ns_dot)

    def test_is_legal(self, engine_context):
        # For a hypothetical 15-cam wheel, we need 8 crosses in D-chi, meaning
        # 4 blocks of cross and 4 of dot.  There must either be 8 cross and 7
        # dot in (un-Delta'd) chi, or 7 cross and 8 dot.
        #
        # Number of ways of choosing a list of four numbers from {1, 2, 3, 4}
        # which add up to 8 is coeff of x^8 in (x + x^2 + x^3 + x^4)^4 = 31
        #
        # Number of ways of choosing a list of four numbers from {1, 2, 3, 4}
        # which add up to 7 is coeff of x^7 in same thing = 20
        #
        # Remove factor of 4 because cycling through the blocks gives equivalent
        # patterns.
        #
        # Then need a factor of two to decide (8 cross, 7 dot) vs vice-versa.
        #
        # Finally an extra factor of 15 to start anywhere.
        #
        # Expected n. legal wheels = 31 * 20 * (1/4) * 2 * 15 = 4,650.

        n_legal = 0
        for i in range(2 ** 15):
            s_chi = '{:015b}'.format(i)
            chi = np.array(list(s_chi), dtype=int)
            wheel = cr.WheelPattern(chi)
            n_legal += wheel.is_legal()

        assert n_legal == 4650


class TestPatterns:
    def test_construction(self):
        cr.Patterns(np.array([1, 0, 1, 1, 0]), np.array([0, 0, 1, 1]))

    def test_construction_wheel_patterns(self):
        wheel1 = cr.WheelPattern(np.array([1, 0, 1, 1, 0]))
        wheel2 = cr.WheelPattern(np.array([0, 0, 1, 1]))
        cr.Patterns(wheel1, wheel2)

    @staticmethod
    def delta_wheel(xs):
        xs_rot1 = np.empty_like(xs)
        xs_rot1[:-1] = xs[1:]
        xs_rot1[-1] = xs[0]
        return xs ^ xs_rot1

    n_cams = [41, 31, 29, 26, 23]

    @pytest.mark.parametrize('n_cams', n_cams)
    def test_random_legal_pattern(self, engine_context, n_cams):
        chi_samples = [self._test_random_legal_pattern_1(engine_context, n_cams)
                       for _ in range(1000)]
        # Allow for birthday paradox:
        assert len(set(chi_samples)) > 800

    def _test_random_legal_pattern_1(self, engine_context, n_cams):
        chi = engine_context.make_legal_wheel_pattern(n_cams)
        assert len(chi) == n_cams
        assert cr.WheelPattern(chi).is_legal()
        return ''.join(map(str, chi))

    def test_is_legal(self, engine_context):
        # Individual wheels' is_legal() tested above, so sanity check only:
        for _ in range(100):
            patterns = engine_context.make_Patterns(41, 31)
            assert patterns.is_legal()

    @pytest.mark.parametrize('n1', n_cams)
    @pytest.mark.parametrize('n2', n_cams)
    def test_random_patterns(self, engine_context, n1, n2):
        patterns = engine_context.make_Patterns(n1, n2)
        assert len(patterns.chi1) == n1
        assert len(patterns.chi2) == n2

    def test_inverted(self, engine_context):
        for _ in range(100):
            patterns = engine_context.make_Patterns(41, 31)
            inv_patterns = patterns.inverted()
            assert np.all(patterns.chi1 != inv_patterns.chi1)
            assert np.all(patterns.chi2 != inv_patterns.chi2)

class TestObservations:
    def test_construction(self):
        cr.Observations(1.25, np.array([[3, 2, 1], [2, 2, 0]], dtype='i'))

    @pytest.mark.parametrize('how', ['chis', 'patterns'])
    def test_random_construction(self, how, engine_context,
                                 sample_chi1, sample_chi2):
        if how == 'chis':
            args = (sample_chi1, sample_chi2, 1.25, 12000)
        elif how == 'patterns':
            patterns = cr.Patterns(sample_chi1, sample_chi2)
            args = (patterns, 1.25, 12000)
        else:
            raise ValueError('unknown how')
        obs = engine_context.make_Observations(*args)
        assert obs.zeta == 1.25
        assert obs.theta.shape == (sample_chi1.size, sample_chi2.size)

    @pytest.mark.parametrize('chi1, chi2',
                             [(zero_chi1(), zero_chi2()),
                              (sample_chi1(), sample_chi2())])
    #
    def test_random_construction_thetas(self, engine_context, chi1, chi2):
        obs = engine_context.make_Observations(chi1, chi2, 3.0, 120000)
        got_theta = obs.theta

        # A zeta of 3.0 means P(dot) = 0.75 and P(cross) = 0.25.  We
        # expect an excess of dot over cross of 0.5, then.  Expected
        # theta values are all around 5000, but not exactly.  Choose
        # values to pass test, but sanity-checking they're near 5000.
        #
        exp_zero_theta = np.array([[5062, 4892, 4922, 4908],
                                   [5004, 4896, 4914, 5042],
                                   [4884, 5070, 5092, 4846]], dtype='i')

        chi12 = chi1[:, None] ^ chi2[None, :]
        assert chi12.shape == got_theta.shape

        chi12_mult = np.where(chi12, -1, 1)
        assert (np.all(chi1 == 0) and np.all(chi2 == 0)) <= np.all(chi12_mult == 1)

        exp_theta = chi12_mult * exp_zero_theta
        nptest.assert_array_equal(obs.theta, exp_theta)

        nptest.assert_allclose(obs.log_zeta_recip, 1.0 / np.log(obs.zeta))

    def test_z_pwr_theta(self, sample_obs):
        nptest.assert_allclose(sample_obs.z_pwr_theta, sample_obs.zeta ** sample_obs.theta)

    def test_zeta_to_power(self, zero_obs):
        for u in np.linspace(-5, 5, 23):
            nptest.assert_allclose(zero_obs.zeta ** u, zero_obs.zeta_to_power(u))

    @pytest.mark.parametrize('row_slc, col_slc',
                             [(slice(None), slice(None)),
                              (slice(None), slice(0, 1)),
                              (slice(0, 1), slice(None))],
                             ids=['matrix', 'col-vec', 'row-vec'])
    #
    def test_chk(self, engine_context, sample_obs, row_slc, col_slc):
        zs = engine_context.unit_normal_shaped_like(sample_obs.theta)[row_slc, col_slc]
        got_chk_value = sample_obs.chk_f(zs)
        exp_chk_value = py_Observations(sample_obs).chk_f(zs)
        nptest.assert_allclose(got_chk_value, exp_chk_value)

class TestDecodingState:
    @staticmethod
    def assert_scores(fgs1, fgs2):
        nptest.assert_allclose(fgs1.score_1, fgs2.score_1)
        nptest.assert_allclose(fgs1.s1, fgs2.s1)
        nptest.assert_array_equal(fgs1.pattern_1, fgs2.pattern_1)
        nptest.assert_allclose(fgs1.score_2, fgs2.score_2)
        nptest.assert_allclose(fgs1.s2, fgs2.s2)
        nptest.assert_array_equal(fgs1.pattern_2, fgs2.pattern_2)

    def state(self, engine_context, sample_obs, label):
        # Create both and then only return the requested one to avoid
        # getting the same values out of the randomness engine.
        rnd_scores_1, rnd_scores_2 = self.random_scores(engine_context, sample_obs)
        sample_acs = self.c_cls(sample_obs, rnd_scores_1, rnd_scores_2)
        c_make_fun = getattr(engine_context, self.c_make_fun_name)
        all_acs = {'sample': sample_acs,
                   'random': c_make_fun(sample_obs)}

        # Check we really did get different scores
        score_1_diff = all_acs['sample'].s1 - all_acs['random'].s1
        assert np.min(np.abs(score_1_diff)) > 0.09

        return all_acs[label]

    def _test_update_score_1(self, engine_context, sample_obs, state_label):
        c_state = self.state(engine_context, sample_obs, state_label)
        py_state = self.py_cls(py_Observations(sample_obs),
                               c_state.score_1, c_state.score_2)

        c_state.update_score_1()
        got_score_1 = c_state.score_1
        py_state = py_state.with_score_1_updated()
        exp_score_1 = py_state.score_1
        nptest.assert_allclose(got_score_1, exp_score_1)
        self.assert_scores(c_state, py_state)

    def _test_update_score_2(self, engine_context, sample_obs, state_label):
        c_state = self.state(engine_context, sample_obs, state_label)
        py_state = self.py_cls(py_Observations(sample_obs),
                               c_state.score_1, c_state.score_2)

        c_state.update_score_2()
        got_score_2 = c_state.score_2
        py_state = py_state.with_score_2_updated()
        exp_score_2 = py_state.score_2
        nptest.assert_allclose(got_score_2, exp_score_2)
        self.assert_scores(c_state, py_state)

    @pytest.mark.parametrize('state_label', ['sample', 'random'])
    def test_convergence(self, engine_context, sample_obs, state_label):
        if type(self) == TestDecodingState:
            return

        c_state = self.state(engine_context, sample_obs, state_label)
        py_state = self.py_cls(py_Observations(sample_obs),
                               c_state.score_1, c_state.score_2)

        converged_p, m_patterns = pr.converge_fg(py_state, 5, 50)
        self.c_converge_fun(c_state, 5, 50)
        #assert False

class TestFactorGraphState(TestDecodingState):
    def test_construction(self, engine_context, sample_obs):
        rnd_scores_1 = engine_context.unit_normal_shaped_like(sample_obs.theta)
        rnd_scores_2 = engine_context.unit_normal_shaped_like(sample_obs.theta)
        fgs = cr.FactorGraphState(sample_obs, rnd_scores_1, rnd_scores_2)
        nptest.assert_array_equal(fgs.score_1, rnd_scores_1)
        nptest.assert_array_equal(fgs.score_2, rnd_scores_2)

    def random_scores(self, engine_context, sample_obs):
        rnd_scores_1 = engine_context.unit_normal_shaped_like(sample_obs.theta)
        rnd_scores_2 = engine_context.unit_normal_shaped_like(sample_obs.theta)
        return rnd_scores_1, rnd_scores_2

    py_cls = pr.FactorGraphState
    c_cls = cr.FactorGraphState
    c_make_fun_name = 'make_FactorGraphState'
    c_converge_fun = cr.converge_FGS

    @pytest.mark.parametrize('state_label', ['sample', 'random'])
    def test_update_score_1(self, engine_context, sample_obs, state_label):
        self._test_update_score_1(engine_context, sample_obs, state_label)

    @pytest.mark.parametrize('state_label', ['sample', 'random'])
    def test_update_score_2(self, engine_context, sample_obs, state_label):
        self._test_update_score_2(engine_context, sample_obs, state_label)

class TestAccurateConvergenceState(TestDecodingState):
    def test_construction(self, engine_context, sample_obs):
        rnd_scores_1 = engine_context.unit_normal_shaped_like(sample_obs.theta)[:, 0]
        rnd_scores_2 = engine_context.unit_normal_shaped_like(sample_obs.theta)[0, :]
        acs = cr.AccurateConvergenceState(sample_obs, rnd_scores_1, rnd_scores_2)
        nptest.assert_array_equal(acs.s1, rnd_scores_1)
        nptest.assert_array_equal(acs.s2, rnd_scores_2)
        nptest.assert_array_equal(acs.pattern_1, (rnd_scores_1 > 0))
        nptest.assert_array_equal(acs.pattern_2, (rnd_scores_2 > 0))

    @staticmethod
    def random_scores(engine_context, sample_obs):
        theta_shape = sample_obs.theta.shape
        rnd_scores_1 = engine_context.unit_normal_of_size(theta_shape[0], 1).ravel()
        rnd_scores_2 = engine_context.unit_normal_of_size(1, theta_shape[1]).ravel()
        return rnd_scores_1, rnd_scores_2

    py_cls = pr.AccurateConvergenceState
    c_cls = cr.AccurateConvergenceState
    c_make_fun_name = 'make_AccurateConvergenceState'
    c_converge_fun = cr.converge_ACS

    @pytest.mark.parametrize('state_label', ['sample', 'random'])
    def test_update_score_1(self, engine_context, sample_obs, state_label):
        self._test_update_score_1(engine_context, sample_obs, state_label)

    @pytest.mark.parametrize('state_label', ['sample', 'random'])
    def test_update_score_2(self, engine_context, sample_obs, state_label):
        self._test_update_score_2(engine_context, sample_obs, state_label)

class TestDirichletState:
    def test_enum(self):
        assert cr.DirichletState.Bound.Lower is not None
        assert cr.DirichletState.Bound.Upper is not None

    def test_direct_construction(self):
        exp_terms = np.array([2, 4, 3, 7], dtype=np.uint32)
        ds = cr.DirichletState(exp_terms, 8)
        assert np.all(ds.terms == exp_terms)

    def parametrize_for_bounds():
        return pytest.mark.parametrize(
            'bnd',
            [cr.DirichletState.Bound.Lower, cr.DirichletState.Bound.Upper],
            ids=['lower', 'upper'])

    @pytest.mark.parametrize('reqd_sum', [10, 11, 12, 13, 20, 21, 22, 23, 24, 30])
    @parametrize_for_bounds()
    def test_bound_construction(self, reqd_sum, bnd):
        ds = cr.DirichletState(10, 4, reqd_sum, bnd)
        raw_terms = ds.terms
        assert np.sum(raw_terms) == reqd_sum

        upper_p = (bnd == cr.DirichletState.Bound.Upper)
        signed_terms = raw_terms.astype(np.int64)
        if upper_p:
            assert np.all(np.diff(signed_terms) <= 0)
        else:
            assert np.all(np.diff(signed_terms) >= 0)

        # This seems a bit perverse but regular expressions capture what
        # we're trying to assert reasonably well:
        upper_terms = (raw_terms if upper_p else raw_terms[::-1])
        terms_as_str = ''.join(map(str, upper_terms))
        assert re.match('^4*[321]?0*$',terms_as_str)

    @parametrize_for_bounds()
    def test_perfect_fit(self, bnd):
        ds = cr.DirichletState(3, 4, 12, bnd)
        assert np.all(ds.terms == 4)

    def test_bad_construction(self):
        bnd = cr.DirichletState.Bound.Lower
        pytest.raises_regexp(ValueError, 'too large',
                             cr.DirichletState, 10, 4, 1000, bnd)

    @pytest.mark.parametrize('max_term', range(5, 9))
    @pytest.mark.parametrize('x_intpart', np.arange(4), ids=str)
    def test_mutation(self, max_term, x_intpart):
        """
        mutate() must:

        Leave all non-selected terms unchanged.
        Leave the pair-sum of the selected two terms unchanged.
        Give (nearly) uniform distribution of results for a fine enough sampling.
        The 'first new term' values should be a contiguous range.
        The 'second new term' values should be a contiguous range.
        Those ranges should be the same.
        For 'big' pair-sums, some term should achieve the maximum.
        For 'small' pair-sums, some term should achieve zero.
        """
        exp_unchd_idxs = sorted(set(range(4)) - set([x_intpart, x_intpart+1]))
        terms_0 = np.array([1, 5, 3, 5, 2], dtype=np.uint64)
        pairs = []
        for x in np.linspace(0.0, 1.0, num=137, endpoint=False):
            ds = cr.DirichletState(terms_0, max_term)
            ds.mutate(x_intpart + x)
            assert np.all(ds.terms[exp_unchd_idxs] == terms_0[exp_unchd_idxs])
            pairs.append(tuple(ds.terms[[x_intpart, x_intpart+1]]))

        exp_pair_sum = terms_0[[x_intpart, x_intpart+1]].sum()
        for pr in pairs:
            assert sum(pr) == exp_pair_sum

        freq = Counter(pairs)
        assert max(freq.values()) - min(freq.values()) <= 1

        all_firsts = np.array(sorted(set(pr[0] for pr in pairs)))
        all_seconds = np.array(sorted(set(pr[1] for pr in pairs)))
        assert np.all(np.diff(all_firsts) == 1)
        assert np.all(all_firsts == all_seconds)

        if exp_pair_sum >= max_term:
            assert np.max(all_firsts) == max_term
        if exp_pair_sum <= max_term:
            assert np.min(all_firsts) == 0

    @pytest.mark.parametrize('n_mutations', [1, 5, 10, 50])
    def test_multiple_mutation(self, n_mutations):
        terms_0 = np.array([1, 5, 3, 5, 2], dtype=np.uint64)
        ds_many = cr.DirichletState(terms_0, 8)
        np.random.seed(42)
        xs = list(np.random.uniform(0.0, 4.0, size=n_mutations))
        ds_many.mutate_many(xs)
        ds_one_by_one = cr.DirichletState(terms_0, 8)
        for x in reversed(xs):
            ds_one_by_one.mutate(x)
        assert np.all(ds_many.terms == ds_one_by_one.terms)


class TestDirichletSamplingState:
    def test_construction(self):
        dss = cr.DirichletSamplingState(10, 3, 12)
        assert dss.maybe_coalesced_result().size == 0
        assert not dss.has_coalesced
        assert dss.lambda_size == 0
        assert dss.n_terms == 10

    def test_always_coalesced(self):
        dss = cr.DirichletSamplingState(10, 3, 30)
        assert np.all(dss.maybe_coalesced_result() == np.full(10, 3, dtype=np.uint64))

    def test_extend_lambda_vec(self):
        dss = cr.DirichletSamplingState(10, 3, 10)
        dss.extend_lambda_vec(np.zeros(7, dtype=np.float64))
        assert not dss.has_coalesced
        assert dss.lambda_size == 7


class TestDirichletSamplingRun:
    def test_construction(self, engine_context):
        dsr = engine_context.make_DirichletSamplingRun(10, 3, 10)
        assert dsr.result.sum() == 10

    @staticmethod
    def two_index_checking(s):
        assert np.min(s) == 2
        assert np.max(s) == 3
        assert np.sum(s) == 29
        return list(s).index(2)

    def test_distribution(self, engine_context):
        """
        The 'known good' values for the final check are just those which
        result from an initial run of the test.  However, they are
        reasonable.  Also, a much larger run (1mln samples) produced the
        counts whose difference from the expected uniform 100000 were

            [-55, 325, 109, -173, -319, 379, -350, -411, 165, 330]

        The following does not account for non-independence between the
        results, but the stddev of a p=0.1 binomial with 100000 draws is
        300, so the above deviations do not seem extreme.
        """
        samples = [engine_context.make_DirichletSamplingRun(10, 3, 29).result
                   for _ in range(1000)]

        two_posns = [self.two_index_checking(s) for s in samples]
        two_posn_freq = Counter(two_posns)
        two_posn_counts = [v for k, v in sorted(two_posn_freq.items())]

        assert two_posn_counts == [91, 97, 106, 83, 98, 107, 109, 92, 106, 111]

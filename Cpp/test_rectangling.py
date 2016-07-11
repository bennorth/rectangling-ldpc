import numpy as np
import numpy.testing as nptest
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

    def test_excess_binomial_rnd_bad_input(self, engine_context):
        pytest.raises_regexp(ValueError, r'must be in \(0, 1\)',
                             engine_context.excess_binomial_rnd, 2, 3, -0.5, 10)

    @pytest.mark.parametrize('p', [0.00001, 0.99999], ids='01')
    @pytest.mark.parametrize('n_leftover', range(6))
    def test_excess_binomial_rnd(self, engine_context, n_leftover, p):
        n_excess = engine_context.excess_binomial_rnd(2, 3, p, 600 + n_leftover)
        raw_exp_n_excess = np.full((2, 3), 100, dtype='i')
        raw_exp_n_excess.ravel()[:n_leftover] += 1
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

class TestPatterns:
    def test_construction(self):
        cr.Patterns(np.array([1, 0, 1, 1, 0]), np.array([0, 0, 1, 1]))

class TestObservations:
    def test_construction(self):
        cr.Observations(1.25, np.array([[3, 2, 1], [2, 2, 0]], dtype='i'))

    def test_random_construction(self, engine_context,
                                 sample_chi1, sample_chi2):
        obs = engine_context.make_Observations(sample_chi1, sample_chi2,
                                               1.25, 12000)
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

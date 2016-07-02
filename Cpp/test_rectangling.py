import numpy as np
import numpy.testing as nptest
import rectangling as cr
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

@pytest.fixture
def sample_chi1():
    return np.array([0, 1, 0], dtype='i')

@pytest.fixture
def sample_chi2():
    return np.array([0, 1, 1, 0], dtype='i')

class TestObservations:
    def test_construction(self):
        cr.Observations(1.25, np.array([[3, 2, 1], [2, 2, 0]], dtype='i'))

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

class TestEngineContext:
    @pytest.fixture
    def ec(self):
        return cr.EngineContext(42)

    def test_construction(self):
        cr.EngineContext(42)

    def test_binomial(self, ec):
        n_successes = ec.test_binomial(0.8, 10000)
        # Value chosen to make test pass, but sanity-checking that it's
        # near 8000.
        assert n_successes == 8029

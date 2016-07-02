import numpy as np
import numpy.testing as nptest
import rectangling as cr

def test_import():
    """
    If we get this far, it was successful.
    """
    assert True

def test_matrix_interchange():
    shape = (7, 13)
    mx = cr.test_matrix_i(*shape)
    assert mx.shape == shape
    assert mx.dtype.kind == 'i'
    nptest.assert_array_equal(mx, 42)

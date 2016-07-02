import numpy as np
from collections import namedtuple

class Observations(namedtuple('Observations', ['zeta', 'theta'])):
    def chk_f(self, x):
        """
        Compute and return matrix of values CHK(theta_ij, x_ij) in the
        'log-base-zeta-likelihood-ratio' representation.  The given
        x should be broadcastable such that x_ij makes sense.
        """
        z = self.zeta
        z_pwr_th = z ** self.theta
        z_pwr_x = z ** x
        lr = (1.0 + z_pwr_th * z_pwr_x) / (z_pwr_th + z_pwr_x)
        return np.log(lr) / np.log(z)

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

class FactorGraphState(namedtuple('FactorGraphState', 'obs score_1 score_2')):
    """
    score_1[i, j] is message sent from K1i to check node linking K1i and K2j (and K12ij).
    score_2[i, j] is message sent from K2j to check node linking K2j and K1i (and K12ij).
    """
    def with_score_1_updated(self):
        """
        Calculate new values for score_1.
        Find Mij message received from K1i along the edge which (via check node involving K12ij) comes from K2j.
        Sum (over j) of these is 'xi'.  Then new score_1[ij] = xi - Mij.
        """
        msgs = self.obs.chk_f(self.score_2)
        msg_sums = np.sum(msgs, axis=1)
        updated_score_1 = msg_sums[:, None] - msgs
        return self._replace(score_1=updated_score_1)

    def with_score_2_updated(self):
        msgs = self.obs.chk_f(self.score_1)
        msg_sums = np.sum(msgs, axis=0)
        updated_score_2 = msg_sums[None, :] - msgs
        return self._replace(score_2=updated_score_2)

    @property
    def s1(self):
        msgs = self.obs.chk_f(self.score_2)
        return np.sum(msgs, axis=1)

    @property
    def s2(self):
        msgs = self.obs.chk_f(self.score_1)
        return np.sum(msgs, axis=0)

    @property
    def pattern_1(self):
        return np.sign(self.s1) == 1.0

    @property
    def pattern_2(self):
        return np.sign(self.s2) == 1.0

class AccurateConvergenceState(namedtuple('AccurateConvergenceState', 'obs score_1 score_2')):
    @property
    def s1(self): return self.score_1

    @property
    def s2(self): return self.score_2

    @property
    def pattern_1(self):
        return np.sign(self.score_1) == 1.0

    @property
    def pattern_2(self):
        return np.sign(self.score_2) == 1.0

    def with_score_1_updated(self):
        score_summands = self.obs.chk_f(self.score_2[None, :])
        updated_score_1 = np.sum(score_summands, axis=1)
        return self._replace(score_1=updated_score_1)

    def with_score_2_updated(self):
        score_summands = self.obs.chk_f(self.score_1[:, None])
        updated_score_2 = np.sum(score_summands, axis=0)
        return self._replace(score_2=updated_score_2)
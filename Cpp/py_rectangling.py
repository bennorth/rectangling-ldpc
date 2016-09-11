# Copyright 2016 Ben North
#
# This file is part of Rectangling-LDPC.
#
# Rectangling-LDPC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Rectangling-LDPC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Rectangling-LDPC.  If not, see <http://www.gnu.org/licenses/>.

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

        Find Mij message received from K1i along the edge which (via
        check node involving K12ij) comes from K2j.

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

def converge_fg(fgs, n_same_converged, max_n_iter):
    prev_patterns = fgs.pattern_1, fgs.pattern_2
    n_same = 1
    n_iterations = 0
    while n_iterations < max_n_iter and n_same < n_same_converged:
        fgs = fgs.with_score_1_updated().with_score_2_updated()
        patterns = fgs.pattern_1, fgs.pattern_2
        if np.all(patterns[0] == prev_patterns[0]) and np.all(patterns[1] == prev_patterns[1]):
            n_same += 1
        else:
            prev_patterns = patterns
            n_same = 1
        n_iterations += 1
    if n_same == n_same_converged:
        return True, patterns
    else:
        return False, None

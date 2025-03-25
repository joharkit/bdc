# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2019 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty5 as ift


class Stokes2Circular(ift.EndomorphicOperator):
    def __init__(self, domain):
        self._domain = ift.DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        if self.domain.shape[0] != 4:
            raise TypeError
        # RR, RL, LR, and LL
        # I Q U V
        # RR = I+V
        # RL = Q+iU
        # LR = Q-iU
        # LL = I-V

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        res = np.empty_like(x, dtype=np.complex128)
        if mode == self.TIMES:
            res[0] = x[0] + x[3]
            res[1] = x[1] + 1j*x[2]
            res[2] = x[1] - 1j*x[2]
            res[3] = x[0] - x[3]
        else:
            res[0] = x[0] + x[3]
            res[1] = x[1] + x[2]
            res[2] = 1j*(-x[1] + x[2])
            res[3] = x[0] - x[3]
        return ift.Field.from_global_data(self._tgt(mode), res)


class Stokes2Linear(ift.EndomorphicOperator):
    def __init__(self, domain):
        self._domain = ift.DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        if self.domain.shape[0] != 4:
            raise TypeError
        # XX, XY, YX, YY
        # I Q U V
        # XX = I+Q
        # XY = U-iV
        # YX = U+iV
        # YY = I-Q

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        res = np.empty_like(x, dtype=np.complex128)
        if mode == self.TIMES:
            res[0] = x[0] + x[1]
            res[1] = x[2] - 1j*x[3]
            res[2] = x[2] + 1j*x[3]
            res[3] = x[0] - x[1]
        else:
            res[0] = x[0] + x[3]
            res[1] = x[0] - x[3]
            res[2] = x[1] + x[2]
            res[3] = 1j*(x[1] - x[2])
        return ift.Field.from_global_data(self._tgt(mode), res)


class Polarization:
    def __init__(self, pol, mode):
        if mode == 'stokesI':
            self._mode = 0
        elif mode == 'stokesIcalibration':
            self._mode = 1
        elif mode == 'polarization':
            self._mode = 2
        else:
            raise ValueError
        self._data_pol = list(pol)
        if set(self._data_pol) <= set([5, 6, 7, 8]):
            self._circular_data = True
        else:
            self._circular_data = False

    # Select polarization
    # RR (5), RL (6), LR (7), and LL (8) for circular polarization
    # XX (9), XY (10), YX (11), and YY (12) for linear polarization

    def sky_extractor(self, label, sky_space):
        if self._mode == 2:
            if label == 'I':
                pos = 0
            elif label == 'Q':
                pos = 1
            elif label == 'U':
                pos = 2
            elif label == 'V':
                pos = 3
            else:
                raise ValueError
            return ift.DomainTupleFieldInserter(sky_space, 0, (pos,)).adjoint
        else:
            return ift.ScalingOperator(1, sky_space)

    def sky2data(self, domain):
        if self._mode == 2 and self._data_pol == [5, 6, 7, 8]:
            return Stokes2Circular(domain)
        if self._mode == 2 and self._data_pol == [9, 10, 11, 12]:
            return Stokes2Linear(domain)
        raise NotImplementedError

    def stokesIdata(self, x):
        if self._circular_data:
            ind = [self._data_pol.index(5), self._data_pol.index(8)]
        else:
            ind = [self._data_pol.index(9), self._data_pol.index(12)]
        return x[ind]

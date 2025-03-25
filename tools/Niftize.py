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
# Copyright(C) 2020 Max-Planck-Society
# Authors: Johannes Harth-Kitzerow

from nifty5 import DomainTuple, from_global_data, LinearOperator
import numpy as np

class Niftize(LinearOperator):
    def __init__(self, domain, target, spMat):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)
        self._spMat = spMat
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        if mode == self.TIMES:
            x = x.to_global_data()
            return from_global_data(self.target, self._spMat.matvec(x))
        else:
            x = x.to_global_data()
            return from_global_data(self.domain, self._spMat.rmatvec(x))

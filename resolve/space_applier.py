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


class SpaceApplier(ift.LinearOperator):
    def __init__(self, op, n):
        n = int(n)
        if not isinstance(op, ift.LinearOperator):
            raise TypeError
        self._domain = ift.DomainTuple.make(
            [ift.UnstructuredDomain(n), *op.domain])
        self._target = ift.DomainTuple.make(
            [ift.UnstructuredDomain(n), *op.target])
        self._capability = op.capability
        self._op = op

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()

        dom = self._op.domain if mode in [
            self.TIMES, self.ADJOINT_INVERSE_TIMES
        ] else self._op.target

        # Handle first entry
        fld = ift.from_global_data(dom, x[0])
        tmp = self._op.apply(fld, mode).to_global_data()
        res = np.empty(self._tgt(mode).shape, dtype=tmp.dtype)
        res[0] = tmp

        # Handle rest
        for ii in range(1, res.shape[0]):
            fld = ift.from_global_data(dom, x[ii])
            res[ii] = self._op.apply(fld, mode).to_global_data()
        return ift.Field.from_global_data(self._tgt(mode), res)

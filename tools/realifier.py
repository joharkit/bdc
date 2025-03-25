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
# Authors: Johannes Harth-Kitzerow, Reimar Leike


import nifty5 as ift
import numpy as np

class Realifier(ift.LinearOperator):
    def __init__(self, domain):
        if not isinstance(domain, ift.UnstructuredDomain):
            raise ValuerError("uyou should not do dis!!1!")
        if not len(domain.shape) == 1:
            raise ValuerError("uyou should not do dis eiser!!1!")
        self._domain = ift.makeDomain(domain)
        tgt = ift.UnstructuredDomain(domain.shape + (2,))
        self._target = ift.makeDomain(tgt)
        self._capability = self._all_ops
    
    def apply(self, x , mode):
        self._check_input(x, mode)
        if (mode == self.TIMES) or (mode == self.ADJOINT_INVERSE_TIMES):
            xval = x.local_data
            res = np.zeros(xval.shape + (2,), dtype=np.float64)
            res[:,0] = xval.real
            res[:,1] = xval.imag
            return ift.from_local_data(self._target, res)
        xval = x.local_data
        res = np.zeros(xval.shape[:-1], dtype=np.complex128)
        res = xval[:,0] + 1.j*xval[:,1]
        return ift.from_local_data(self._domain, res)


if __name__ == "__main__":
    sp = ift.UnstructuredDomain(20)
    R = Realifier(sp)
    x = ift.from_random('normal', sp) + 1.j*ift.from_random('normal', sp)
    y = ift.from_random('normal', R.target)
    print((R.inverse(R(x))-x).norm())

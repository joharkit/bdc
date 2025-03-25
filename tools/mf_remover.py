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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import absolute_import, division, print_function

from nifty5 import DomainTuple, UnstructuredDomain, from_global_data, LinearOperator, MultiField

from numpy import zeros, array, cumsum


class MFRemover(LinearOperator):
    """Operator which transforms between a structured MultiDomain
    and an unstructured domain.

    Parameters
    ----------
    domain: MultiDomain
        the full input domain of the operator.

    Notes
    -----
    The operator will convert the full domain of its input domain to an
    UnstructuredDomain 
    """

    def __init__(self, domain):
        self._domain = domain
        self._size_array = array([0]+[d.size for d in domain.values()])
        cumsum(self._size_array, out=self._size_array)
        target = UnstructuredDomain(self._size_array[-1])
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        if mode == self.TIMES:
            res = zeros(self.target.shape)
        else:
            res = {}
            x_val = x.to_global_data()
        i = 0
        for k in self.domain.keys():
            if mode == self.TIMES:
                res[self._size_array[i]:self._size_array[i+1]] =\
                    x[k].to_global_data().reshape(-1)
            else:
                res[k] = from_global_data(self.domain[k], 
                        x_val[self._size_array[i]:self._size_array[i+1]].reshape(self.domain[k].shape))
            i += 1
        if mode == self.TIMES:
            return from_global_data(self.target, res)
        else:
            return MultiField.from_dict(res)



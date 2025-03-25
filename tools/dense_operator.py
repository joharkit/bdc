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
# Copyright(C) 2013-2020 Max-Planck-Society


from nifty5 import LinearOperator, from_global_data, UnstructuredDomain, DomainTuple
from numpy import prod

class DenseOperator(LinearOperator):
    
    def __init__(self, domain, matrix):
        """
        Operator that applies a dense matrix

        Parameters
        ----------
        domain: Domain
            domain of the input. can be anything with a shape
        matrix: ndarray
            matrix to be applied. should be of shape 
            (a1,...,an, b1,...,bm), where (b1,..,bm) is the shape of
            the domain and (a1,...,an) is the shape of the target
        """
        self._domain = DomainTuple.make(domain)
        in_shape = domain.shape
        ndim = len(in_shape)
        if not (matrix.shape[-ndim:] == in_shape):
            raise ValueError("Matrix shape is incompatable with domain")
        out_shape = matrix.shape[:-ndim]
        self._target = DomainTuple.make(UnstructuredDomain(out_shape))
        self._matrix = matrix.reshape((prod(out_shape), prod(in_shape)))
        self._matrix.flags.writeable = False
        self._capability = self.TIMES | self.ADJOINT_TIMES
    

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            out = self._matrix @ x.to_global_data()
            return from_global_data(self.target, 
                    out.reshape(self.target.shape))
        if mode == self.ADJOINT_TIMES:
            out = x.to_global_data() @ self._matrix 
            return from_global_data(self.domain, 
                    out.reshape(self.domain.shape))


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


from __future__ import absolute_import, division, print_function

from nifty5 import DomainTuple, from_global_data, LinearOperator

from numpy import prod, einsum


class Reshaper(LinearOperator):
    """
    Operator that reshapes a field as in numpy.reshape
    
    Parameters
    ----------
    domain: Domain
    target: Target
      should be of same size as Domain
    """

    def __init__(self, domain, target):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)
        if domain.size != target.size:
            raise ValueError("Incompatible shapes!")
        self._capability = 15 # self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES or mode == self.ADJOINT_INVERSE_TIMES:
            x = x.to_global_data()
            return from_global_data(self.target, x.reshape(self.target.shape))
        else:
            x = x.to_global_data()
            return from_global_data(self.domain, x.reshape(self.domain.shape))

class Counter(LinearOperator):
    """
    Operator that counts the number of calls
    
    Parameters
    ----------
    domain: Domain
      should be of same size as Domain
    """

    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(domain)
        self._capability = 15 # self.TIMES | self.ADJOINT_TIMES
        self.Rcounter=0

    def apply(self, x, mode):
        self._check_input(x, mode)
        self.Rcounter+=1
        if mode == self.TIMES or mode == self.ADJOINT_INVERSE_TIMES:
            return x
        else:
            return x

class Matmul(LinearOperator):
    """
    Operator that implements matrix multiplication
    
    Parameters
    ----------
    domain: Domain
    target: Target
      should be of same size as Domain
    """

    def __init__(self, domain, target, np_matrices):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._mat = np_matrices

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            x = x.to_global_data()
            return from_global_data(self.target, einsum("ijk,ik->ij",self._mat, x))
        else:
            x = x.to_global_data()
            return from_global_data(self.domain,  einsum("ijk,ij->ik",self._mat, x))

class Transposer(LinearOperator):
    """
    Permute the dimensions of an array
    Parameters
    ----------
    domain: Domain
    target: Target
      should be of same size as domain
    transposearray: Array, that specifies the transposition
      should have same length as domain.shape and target.shape
    """

    def __init__(self, domain, target, transposearray):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)
        self._transposearray = transposearray
        if domain.size != target.size:
            raise ValueError("Incompatible shapes: Domain and Target are supposed to have the same size")
        self._capability = 15 # self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        transposearray = self._transposearray
        if len(x.shape) != len(transposearray):
            raise ValueError("Incompatible shapes: Length of x.shape and transposearray are not the same")
        if mode == self.TIMES or mode == self.ADJOINT_INVERSE_TIMES:
            x = x.to_global_data()
            return from_global_data(self.target, x.transpose(transposearray))
        else:
            x = x.to_global_data()
            return from_global_data(self.domain, x.transpose(transposearray))

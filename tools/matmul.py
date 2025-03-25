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
# Author: Philipp Frank


import numpy as np
import nifty5 as ift


class AddAxis(ift.LinearOperator):
    def __init__(self,domain,ind):
        self._domain = ift.DomainTuple.make(domain)
        if ind<0:
            ind = len(self.domain)+ind +1
        tgt = self.domain[:ind] + (ift.UnstructuredDomain(1),) + self.domain[ind:]
        self._target = ift.DomainTuple.make(tgt)
        self._ind = 0
        for d in self.domain[:ind]:
            self._ind += len(d.shape)
        self._capability = self.TIMES | self.ADJOINT_TIMES
    def apply(self,x,mode):
        self._check_input(x,mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            res = np.expand_dims(x,self._ind)
        else:
            res = x.sum(axis=self._ind)
        return ift.from_global_data(self._tgt(mode),res)

class Hat(ift.LinearOperator):
    def __init__(self,domain):
        self._domain = ift.DomainTuple.make(domain)
        tmp = ift.DomainTuple.make(self.domain[:-1])
        tgt = self.domain[:-1]+(self.domain[-1],self.domain[-1])
        self._target = ift.DomainTuple.make(tgt)
        ind = np.indices(self.domain[-1].shape)
        self._domsl = (slice(None),)*len(tmp.shape) + (ind,)
        self._tarsl = (slice(None),)*len(tmp.shape) + (ind,ind)
        self._capability = self.TIMES | self.ADJOINT_TIMES
    def apply(self,x,mode):
        self._check_input(x,mode)
        x = x.to_global_data()
        res = np.zeros(self._tgt(mode).shape)
        if mode == self.TIMES:
            res[self._tarsl] = x[self._domsl]
        else:
            res[self._domsl] = x[self._tarsl]
        return ift.from_global_data(self._tgt(mode),res)

class Expander(ift.LinearOperator):
    def __init__(self,domain,newspace):
        self._domain = ift.DomainTuple.make(domain)
        if isinstance(newspace,ift.DomainTuple):
            tn = newspace[:]
        else:
            tn = (newspace,)
        self._target = ift.DomainTuple.make(tn+self.domain[:])
        self._inds = tuple(np.arange(len(newspace.shape)))
        self._addshp = newspace.shape
        self._capability = self.TIMES | self.ADJOINT_TIMES
    def apply(self,x,mode):
        self._check_input(x,mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            res = np.tile(x,self._addshp + len(self.domain.shape)*(1,))
        else:
            res = x.sum(axis=self._inds)
        return ift.from_global_data(self._tgt(mode),res)

class MatConjugator(ift.LinearOperator):
    def __init__(self,domain):
        self._domain = ift.DomainTuple.make(domain)
        tgt = self._domain[:-2]+(self._domain[-1],self._domain[-2],)
        self._target = ift.DomainTuple.make(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self,x,mode):
        self._check_input(x,mode)
        res = np.swapaxes(x.to_global_data(),-1,-2)
        return ift.from_global_data(self._tgt(mode),res)

class StaticMatMul(ift.LinearOperator):
    def __init__(self,M,domain,target):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._M = M
        self._capability = self.TIMES | self.ADJOINT_TIMES
        
    def apply(self,x,mode):
        self._check_input(x,mode)
        x = x.to_global_data()
        if mode==self.TIMES:
            res = np.matmul(self._M,x)
        else:
            res = np.matmul(np.swapaxes(self._M,-1,-2),x)
        return ift.from_global_data(self._tgt(mode),res)

class VariableMatMul(ift.Operator):
    def __init__(self,A,B):
        self._domain = ift.domain_union((A.domain,B.domain))
        tgt = A.target[:-1]+(B.target[-1],)
        self._target = ift.DomainTuple.make(tgt)
        self._A,self._B = A,B

    def apply(self,x):
        self._check_input(x)
        lin = isinstance(x, ift.Linearization)
        v = x._val if lin else x
        vA = v.extract(self._A.domain)
        vB = v.extract(self._B.domain)
        if not lin:
            A = self._A(vA)
            B = self._B(vB)
            res = StaticMatMul(A.to_global_data(),B.domain,
                               self._target)(B)
            return res
        wm = x.want_metric
        linA = self._A(ift.Linearization.make_var(vA, wm))
        linB = self._B(ift.Linearization.make_var(vB, wm))
        mulA = StaticMatMul(linA.val.to_global_data(),
                            linB.val.domain,self._target)
        res = mulA(linB.val)
        op = MatConjugator(linA.val.domain)(linA.jac)
        con = MatConjugator(self._target)
        op = StaticMatMul(linB.val.to_global_data(),
                          con.target,op.target).adjoint(op)
        op = (mulA(linB.jac) + con.adjoint(op))
        return x.new(res,op(x.jac))



if __name__ == '__main__':
    dom1 = ift.RGSpace((10,10))
    dom2 = ift.UnstructuredDomain(3)
    dom22 = ift.UnstructuredDomain(4)
    dom23 = ift.UnstructuredDomain(1)
    
    dom = ift.DomainTuple.make((dom1,dom2,dom22))
    domb = ift.DomainTuple.make((dom1,dom22,dom23))
    a = ift.FieldAdapter(dom,'a')
    b = ift.FieldAdapter(domb,'b')
    
    op = VariableMatMul(a,b)
    pos = ift.from_random('normal',op.domain)
    
    oplin = op(ift.Linearization.make_var(pos,True))
    
    ift.extra.consistency_check(Expander(dom,dom1))
    ift.extra.consistency_check(Hat(dom))
    ift.extra.consistency_check(AddAxis(dom,1))
    
    ift.extra.consistency_check(oplin.jac)
    
    ift.extra.check_jacobian_consistency(op,pos)
    
    

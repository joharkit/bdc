'''
Parts of the code are from
https://wendynavarrete.com/principal-component-analysis-with-numpy/
online accessed on 2020-11-19
'''

import numpy as np
import nifty5 as ift

from sklearn.decomposition import PCA
from tools.reshaper import Reshaper
from bdc.Compress import LOp
from tools.dense_operator import DenseOperator
from scipy.sparse.linalg import eigs, LinearOperator

def bayesian_pca(R, S, N,
        kmax = None,
        numpy = False):
    RSR = ift.SandwichOperator.make(R.adjoint, S)
    data_cov = RSR + N
    l, v = eigs(LOp(data_cov), k=kmax)
    l = l.real
    v = v.real
    try:
        N_np = np.diag(N._ldiag)
    except:
        raise NotImplementedError("N not diagonal")
    N_c = v.T @ N_np @ v
    l_n, v_n = np.linalg.eigh(N_c)
    l_n_inv = 1./np.sqrt(l_n)
    sqrtN_inv = DenseOperator(ift.UnstructuredDomain(N_c.shape[0]), (v_n *
        l_n_inv).T)
    N_inv = ift.SandwichOperator.make(sqrtN_inv)
    if numpy:
        return v.T, N_inv
    else:
        pca_trafo = DenseOperator(N.domain, v.T)
        return pca_trafo, N_inv


class pca(ift.LinearOperator):
    def __init__(self, d_o, kmax):
        self._domain = d_o.domain
        self._d_o = d_o
        self._kmax = kmax
        self._pca = PCA(n_components=self._kmax)
        self._pca.fit(self._d_o.to_global_data())
        d_c = self._pca.transform(self._d_o.to_global_data())
        self._target = ift.UnstructuredDomain(d_c.shape)
        self.d_c = ift.from_global_data(self.target, d_c)
        self.N_c_inv = ift.DenseOperator(self.target, self._pca.get_precision())

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES or mode == self.ADJOINT_INVERSE_TIMES:
            x = x.to_global_data()
            return ift.from_global_data(self.target, self._pca.transform(x))
        else:
            x = x.to_global_data()
            return ift.from_global_data(self.domain, self._pca.inverse_transform(x))

def standard_pca(d_o, R_o, kmax):
    Reshape = Reshaper(d_o.domain,
            ift.UnstructuredDomain((d_o.domain.shape+(1,))))
    d_o = Reshape(d_o)
    pca_trafo = pca(d_o, kmax)
    d_c = pca_trafo.d_c
    N_c_inv = pca_trafo.N_c_inv
    R_c = pca_trafo @ Reshape @ R_o
    return d_c, R_c, N_c_inv

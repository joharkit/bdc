#!/usr/bin/python3
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


import numpy as np
import nifty5 as ift
from scipy.sparse.linalg import eigs, LinearOperator
from tools.dense_operator import DenseOperator
from tools.mf_remover import MFRemover


# Input: NIFTy Operator
# Output: Numpy representation of Op at place x
def numpyze(Op):
    in_shape = Op.domain.shape
    return(lambda x: Op(ift.from_global_data(Op.domain, x.reshape(in_shape))).to_global_data().reshape(-1)) #from_global_data: np -> ift, to_global_data: ift -> np

# Make Op a linear operator
# !No Multifield!
def LOp(Op):
    dim_in = np.prod(Op.domain.shape)
    dim_out = np.prod(Op.target.shape)
    return LinearOperator(dtype=np.float64, shape=(dim_out, dim_in), matvec=numpyze(Op))

##################
# Nonlinear Case #
##################

def Linearize(mean,signal,Mo,jo,
        samps=None,
        IC=ift.GradientNormController(iteration_limit = 500, tol_abs_gradnorm =1e-3)):
    if samps== None:
        lin = ift.Linearization.make_var(mean)
        jac = signal(lin).jac
        S = ift.SandwichOperator.make(jac.adjoint)
    else:
        S = ift.ScalingOperator(signal.target, 0.)
        for samp in samps:
            lin = ift.Linearization.make_var(mean + samp)
            jac = signal(lin).jac
            S += ift.SandwichOperator.make(jac.adjoint)
        S.scale(1/len(samps))
    #Transform Operators to Excitation Space (with \sqrt{S})
    calDo_inv = ift.SandwichOperator.make(jac, Mo)+ift.ScalingOperator(1., signal.domain)
    calDo = ift.InversionEnabler(calDo_inv, IC).inverse
    caljo = jac.adjoint(jo)
    calmo = calDo(caljo)
    #and back
    mo = jac(calmo) 
    return mo, S

#######
# BDC #
#######

def DeltaIexact(musqd, r, mo):
    # Throw away too few informative eigenpairs
    deltasqd = 1./(musqd+1.)
    rmmr = [np.vdot(r[:,ii],mo.to_global_data())*np.vdot(mo.to_global_data(),r[:,ii]) for ii in range(len(r[0]))]
    DeltaI = 0.5*(deltasqd-1.-np.log(deltasqd)+rmmr)
    return DeltaI

def DeltaIexp(musqd):
    return 0.5*np.log(musqd+1.)

def compress(mo, S, Mo, kmax, Ro=None, gamma = 0.7,
        multifield=False,
        IC=ift.GradientNormController(iteration_limit = 500, tol_abs_gradnorm =1e-3),
        numpy=False,
        k_gamma = None):
    '''Compress linear measurement parameters: data, response and noise

    Parameters:
    ----------
    mo  :   posterior mean
    S   :   prior covariance
    Mo  :   R.T N_inv R
    kmax:   maximum dimension of the compressed data
    Ro  :   original response
    gamma:  CompressionFactor; The minimum fraction of information to be stored in the
            compressed data
    multifield: boolean for multifield input
    IC  :   Iteration norm controller
    numpy:  boolean whether output shall be numpy, else NIFTy object
    k_gamma:Number of eigenpairs determined by the algorithm

    Returns:
    -------
    d_c : compressed data
    R_c : compressed response
    N_c_inv " compressed noise covariance diagonal
    gamma : (relative) amount of information stored in the compressed data as
    a tuple with lower and higher boundary of the estimation
    '''
    if kmax == 0:
        return
    if k_gamma is None:
        k_gamma = kmax
    MoS = Mo(S)
    #Numpyze:
    npS = LOp(S)
    npMoS = LOp(MoS)
    
    musqd, r = eigs(npMoS, k=k_gamma)
    musqd = musqd.real
    r = r.real

    #Normalize eigenvectors w.r.t. to <,>_S:
    for i in range(len(r[0])):
        r[:,i] = r[:,i]/(np.sqrt(np.abs(np.vdot(r[:,i],npS(r[:,i])))))
    
    DeltaI = DeltaIexp(musqd)
    if Ro is not None:
        K = min(Ro.target.size, Ro.domain.size)
    else:
        K = npMoS.shape[0]
    criterion = (1-gamma)/gamma*np.cumsum(DeltaI)/(K-np.arange(len(DeltaI))-1)
    gamma_min = np.sum(DeltaI[0:(min(len(DeltaI),kmax)-1)])/(np.sum(DeltaI)+(K-len(DeltaI))*DeltaI[-1])
    gamma_max = np.sum(DeltaI[0:(min(len(DeltaI),kmax)-1)])/(np.sum(DeltaI))
    print("gamma >= {}".format(gamma_min))
    if DeltaI[-1] > criterion[-1]:
        print("compressed data might contain less than "+str(gamma)+" of the total information.")
    else:
        print("Through away "+str(len(musqd)-np.sum(DeltaI > criterion))+" eigendirections")
        musqd = musqd[DeltaI > criterion]
        r = r[:,DeltaI > criterion]
    musqd = musqd[:min(kmax, len(musqd))]
    r = r[:,:min(kmax, len(musqd))]
    
    if multifield:
        FA = ift.FieldAdapter(S.domain, 'DummyField').adjoint
        mfr = MFRemover(FA.target)
        denseOp = DenseOperator(ift.UnstructuredDomain(Mo.domain.size), r.T)
        Rc = denseOp @ mfr @ FA
    else:
        Rc = DenseOperator(S.domain, r.T)
    Nc_inv_diag = ift.from_global_data(Rc.target, musqd)
    Nc_inv = ift.DiagonalOperator(Nc_inv_diag)
    
    RcSRc = ift.SandwichOperator.make(Rc.adjoint, S)
    RcSRc_inv = ift.InversionEnabler(RcSRc, IC).inverse
    dc = (Nc_inv.inverse + RcSRc) (RcSRc_inv(Rc(mo)))
    if not numpy:
        return dc, Rc, Nc_inv, [gamma_min, gamma_max]
    return dc.to_global_data(), r.T, musqd, [gamma_min,gamma_max]

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
#For detailed explanation, see NIFTy Demos getting_started_1 & getting_started_3


#For creating directories
import os
import pickle
import numpy as np
import nifty5 as ift
import matplotlib
# For png
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, eigsh, LinearOperator
from tools.mf_remover import MFRemover
from tools.reshaper import Reshaper, Transposer, Matmul, Counter
from tools.matmul import StaticMatMul
from tools.SamplingEnabler import SamplingEnabler

import time

from bdc.Compress import compress, Linearize



def make_checkerboard_mask(position_space):
    # Checkerboard mask for 2D mode
    mask = np.ones(position_space.shape)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                mask[i*128//4:(i + 1)*128//4, j*128//4:(j + 1)*128//4] = 0
    return mask


    
    ##### PIPER #####
    
    

def RGCheckerboardPatcher(domain, patch_number):
    patch_domain = ift.UnstructuredDomain([patch_number[0],domain.shape[0]//patch_number[0],patch_number[1],domain.shape[1]//patch_number[1]])
    rsp1 = Reshaper(domain, patch_domain)
    transposearray = (0,2,1,3)
    trsp = Transposer(patch_domain, ift.UnstructuredDomain([patch_number[0],patch_number[1], domain.shape[0]//patch_number[0], 
                                                            domain.shape[1]//patch_number[1]]), transposearray)
    tgt = ift.DomainTuple.make([ift.UnstructuredDomain(np.prod(patch_number)), ift.RGSpace([domain.shape[0]//patch_number[0],domain.shape[1]//patch_number[1]])])
    rsp2 = Reshaper(trsp.target, tgt)
    checkp = rsp2 @ trsp @ rsp1
    return checkp


def UDCheckerboardPatcher(domain, patch_number):
    patch_domain = ift.UnstructuredDomain([patch_number[0],domain.shape[0]//patch_number[0],patch_number[1],domain.shape[1]//patch_number[1]])
    rsp1 = Reshaper(domain, patch_domain)
    transposearray = (0,2,1,3)
    trsp = Transposer(patch_domain, ift.UnstructuredDomain([patch_number[0],patch_number[1], domain.shape[0]//patch_number[0], 
                                                            domain.shape[1]//patch_number[1]]), transposearray)
    rsp2 = Reshaper(trsp.target, ift.UnstructuredDomain([np.prod(patch_number),domain.shape[0]//patch_number[0],domain.shape[1]//patch_number[1]]))
    checkp = rsp2 @ trsp @ rsp1
    return checkp



if __name__ == '__main__':
    np.random.seed(420)
    jobname = int(time.time())
    # maximum number of compressed data points per patch
    kmax = 10
    # number of eigenpairs determined to better estimate gamma
    # should be greater or equal kmax
    k_gamma = kmax

    # compression factor
    gamma = 0.99
    print("gamma = {}".format(gamma))

    # number of samples used to estimate the KL
    N_samples = 20

    # Draw new samples to approximate the KL n_rep times
    n_rep = 2
    n_comp = 3
    n_o = n_comp*n_rep
    
    # dimensions of position space
    ps_dim = np.array([128,128])
    # Patching
    patch_number = np.array([4,4])
    patches_number = np.prod(patch_number)
    
    kmax_c= kmax*patches_number//2
    k_gamma_c = k_gamma*patches_number//2
    
    path = "Wimmerl/{}".format(jobname)

    try:
        os.makedirs("Wimmerl", mode=0o777)
        print("created directory Wimmerl")
    except:
        pass
    try:
        os.makedirs(path, mode=0o777)
        print("created directory "+path)
    except:
        pass
    try:
        os.makedirs("./Plots", mode=0o777)
        print("created directory Plots")
    except:
        pass


 
    filename = "./Plots/bdc_nl_CompTo_{}_datapoints_".format(kmax_c)+"{}_patches".format(patches_number)+"_{}_pP".format(kmax)+"_{}_".format(N_samples)+"{}_".format(n_rep)+"{}_".format(n_comp)+"{}.pdf"
    with open(path+'/nl_setup.txt','w') as f:
        f.writelines(["kmax_c = {}\n".format(kmax_c),
            "gamma = {}\n".format(gamma),
            "kmax = {}\n".format(kmax),
            "k_gamma = {}\n".format(k_gamma),
            "patches_number = {}\n".format(patches_number),
            "N_samples = {}\n".format(N_samples),
            "n_rep = {}\n".format(n_rep),
            "n_comp = {}".format(n_comp)]
            )


    ##############
    ###1. Setup###
    ##############

    position_space = ift.RGSpace(ps_dim)
    mask = make_checkerboard_mask(position_space)
    harmonic_space = position_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(harmonic_space, target=position_space)
    power_space = ift.PowerSpace(harmonic_space)
    
    # Set up an amplitude operator for the field
    dct = {
        'target': power_space,
        'n_pix': 64,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)262
        'a': 3,  # relatively high variance of spectral curvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im':  0,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3   # y-intercept variance
    }
    A = ift.SLAmplitude(**dct)

    # Build the operator for a correlated signal
    power_distributor = ift.PowerDistributor(harmonic_space, power_space)
    vol = harmonic_space.scalar_dvol**-0.5
    xi = ift.ducktape(harmonic_space, None, 'xi')
    correlated_field = ht(vol*power_distributor(A)*xi)

    # Apply a sigmoid function on the correlated field
    signal = ift.sigmoid(correlated_field)
    
    
    # Data is defined on a geometry-free space, thus the geometry is removed
    GR = ift.GeometryRemover(position_space)

    # Masking operator to model that parts of the field have not been observed
    mask = ift.Field.from_global_data(position_space, mask)
    Mask = ift.DiagonalOperator(mask)  
    

    CntR = Counter(GR.target)
    # The response operator consists of
    # - an harmonic transform (to get to image space)
    # - the application of the mask
    # - the removal of geometric information
    # Operators can be composed either with parenthesis
    R = CntR @ GR @ Mask
    
    
    signal_response_o = R(signal)
    
     
    # Specify noise
    data_space = R.target
    noise_std = .02 #5.
    N = ift.ScalingOperator(noise_std**2, data_space)
    Ninv = N.inverse

    # Generate mock signal and data
    mock_position = ift.from_random('normal', signal_response_o.domain)
    data = signal_response_o(mock_position) + N.draw_sample()
    

    # Minimization parameters
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradInfNormController(
        name='Newton', tol=1e-7, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)

    # Set up likelihood and information Hamiltonian
    likelihood = ift.GaussianEnergy(mean=data, inverse_covariance=Ninv)(signal_response_o)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean
    
    with open(path+'/setup.pk', 'wb') as f:
        pickle.dump([filename, signal, A, mock_position, patch_number], f) 



    #################
    ###Full Problem##
    #################
    
    
    print("Now doing usual inference")
    
    CntR.Rcounter = 0
    sumtime_o = time.perf_counter()
    mean_o = initial_mean
    likelihoodo = ift.GaussianEnergy(mean=data, inverse_covariance=Ninv)(signal_response_o)
    Ho = ift.StandardHamiltonian(likelihoodo, ic_sampling)
    
    inftime_o = []
    for ll in range(n_comp):
        inftime_o.append(-time.perf_counter())
        for ii in range(n_rep):
            # Draw new samples and minimize KL
            KL_o = ift.MetricGaussianKL(mean_o, Ho, N_samples, mirror_samples=True)
            KL_o, convergenceo = minimizer(KL_o)
            mean_o = KL_o.position
            print((ll*n_rep+ii)*100/(n_o))
       
            likelihoodo = ift.GaussianEnergy(mean=data, inverse_covariance=Ninv)(signal_response_o)
            Ho = ift.StandardHamiltonian(likelihoodo, ic_sampling)
        inftime_o[-1] += time.perf_counter()
    
    sumtime_o = time.perf_counter() - sumtime_o


    # Draw posterior samples
    KL_o = ift.MetricGaussianKL(mean_o, Ho, N_samples)
    sco = ift.StatCalculator()
    for sample in KL_o.samples:
        sco.add(signal(sample + KL_o.position))
    
    with open(path+'/rcounter.txt','a') as f:
        f.write("Full: Rcounter = {}\n".format(CntR.Rcounter)
            )
      
    with open(path+"/NonLin_o.pk",'wb') as f:
          pickle.dump([KL_o,R,Ninv,data,sco.mean,sco.var, inftime_o, sumtime_o], f)
    
    
    #################
    ###Compression###
    #################
    print("Compression")
    
    
    filename = "./Plots/bdc_nonLinCheck_CompTo_{}_datapoints_".format(kmax_c)+"and_{}_patches".format(patches_number)+"_{}_".format(N_samples)+"{}_".format(n_rep)+"{}_".format(n_comp)+"{}.pdf"
    
    Mo = ift.SandwichOperator.make(R,Ninv)
    CntR.Rcounter=0
    
    
    sumtime_c = time.perf_counter()
    mean_c = initial_mean
    jo = R.adjoint_times(N.inverse_times(data))
    m_temp, S_temp = Linearize(mean_c, signal, Mo, jo)
    d_c, R_c, N_c_inv, gamma_min_c = compress(m_temp, S_temp, Mo, kmax_c, Ro=R,
            gamma=gamma, multifield=True, k_gamma=k_gamma_c)
    R_c0 = R_c
    d_c0 = d_c
    
    
    signal_responsec = R_c(signal)
    
    likelihood_c = ift.GaussianEnergy(mean=d_c, inverse_covariance=N_c_inv)(signal_responsec)
    Hc = ift.StandardHamiltonian(likelihood_c, ic_sampling)
    
    ###################
    #Comp Minimization#
    ###################
    inftime_c = []
    for l in range(n_comp):
        inftime_c.append(-time.perf_counter())
        for i in range(n_rep):
            # Draw new samples and minimize KL_c
            KL_c = ift.MetricGaussianKL(mean_c, Hc, N_samples, mirror_samples=True)
            KL_c, convergence = minimizer(KL_c)
            mean_c = KL_c.position
            print((l*n_rep+i+1)*100/(n_comp*n_rep))
        inftime_c[-1] += time.perf_counter()

        if l < n_comp-1:
            comptime_c = time.perf_counter()
            m_temp, S_temp = Linearize(mean_c, signal, Mo, jo)
            d_c, R_c, N_c_inv, gamma_min_c = compress(m_temp, S_temp, Mo,
                    kmax_c, Ro=R, gamma=gamma, multifield=True,
                    k_gamma=k_gamma_c)
            signal_responsec = R_c(signal)
            
            likelihood_c = ift.GaussianEnergy(mean=d_c, inverse_covariance=N_c_inv)(signal_responsec)
            Hc = ift.StandardHamiltonian(likelihood_c, ic_sampling)
            comptime_c = time.perf_counter() - comptime_c
            
            # Convergence criteria
            sumtime_c -= time.perf_counter()
            # Draw posterior samples
            KL_c = ift.MetricGaussianKL(mean_c, Hc, N_samples)
            sc = ift.StatCalculator()
            for sample in KL_c.samples:
                sc.add(signal(sample + KL_c.position))
            IC=ift.GradientNormController(iteration_limit = 500, tol_abs_gradnorm =1e-3)
            lin = ift.Linearization.make_var(mean_c)
            jac = signal(lin).jac
            S_c = ift.SandwichOperator.make(jac.adjoint)
            with open(path+"/NonLin_c_{}.pk".format(l),'wb') as f:
                  pickle.dump([S_c, R_c, N_c_inv, d_c, sc.mean], f)
            sumtime_c += time.perf_counter()
    
    sumtime_c = time.perf_counter() - sumtime_c


    # Draw posterior samples
    KL_c = ift.MetricGaussianKL(mean_c, Hc, N_samples)
    sc = ift.StatCalculator()
    for sample in KL_c.samples:
        sc.add(signal(sample + KL_c.position))
    
    IC=ift.GradientNormController(iteration_limit = 500, tol_abs_gradnorm =1e-3)
    lin = ift.Linearization.make_var(mean_c)
    jac = signal(lin).jac
    S_c = ift.SandwichOperator.make(jac.adjoint)

    # Store Variance in Response
    S_r = ift.InversionEnabler(S_c,IC)
    S_r = SamplingEnabler(S_r,IC)
    D_r_inv = ift.WienerFilterCurvature(R_c,N_c_inv.inverse,S_r,IC,IC)

    sc_r = ift.StatCalculator()
    for sample in range(N_samples):
        sc_r.add(D_r_inv.draw_sample(from_inverse=True))
    unc = sc_r.var

    with open(path+'/rcounter.txt','a') as f:
        f.write("Comp: Rcounter = {}\n".format(CntR.Rcounter)
            )
    with open(path+"/NonLin_c.pk",'wb') as f:
          pickle.dump([KL_c, S_c, R_c, N_c_inv, d_c, sc.mean, sc.var, R_c0,
              d_c0, unc, gamma_min_c, inftime_c, sumtime_c], f)
    
      
      
    ##############
    ###PATCHING###
    ##############
    
    print("cutting into patches")
    
    filename = "./Plots/bdc_nonLinCheck_CompTo_{}_datapoints_pP".format(kmax)+"and_{}_patches".format(patches_number)+"_{}_".format(N_samples)+"{}_".format(n_rep)+"{}_".format(n_comp)+"{}.pdf"
    
    inftime_pc = []
    sumtime_pc = time.perf_counter()
    patch_dim = ps_dim//patch_number
    GR_p = ift.GeometryRemover(ift.RGSpace(patch_dim))
    CntR = Counter(GR_p.target)
    Rp = CntR @ GR_p
    
    CntR.Rcounter=0
    mean_pc = initial_mean
    isvalidpatch = [1,3,4,6,9,11,12,14]
    Np = ift.ScalingOperator(noise_std**2, Rp.target)
    Npinv = Np.inverse
    Mop = ift.SandwichOperator.make(Rp,Npinv)
    jop = [[] for i in range(patches_number)]
    d_pc  = np.zeros((patches_number, kmax))
    R_pc  = np.zeros((patches_number, kmax, np.prod(patch_dim)))
    Npc_inv = np.zeros((patches_number, kmax))
    gamma_min_pc = [[0.,1.]]*patches_number
    
    npdata = UDCheckerboardPatcher(data.domain, patch_number)(data).to_global_data()
    RGCP = RGCheckerboardPatcher(signal.target, patch_number)
    patched_signal = RGCP(signal)
    
    comptime_pc = time.perf_counter()
    for i in range(patches_number):
        print(100*i/patches_number)
        if i in isvalidpatch:
            selector = ift.DomainTupleFieldInserter(patched_signal.target, 0, (i,)).adjoint
            signalp = selector @ patched_signal
            datap = ift.from_global_data(ift.UnstructuredDomain(npdata[i].shape), npdata[i])
            jop[i] = Rp.adjoint_times(Np.inverse_times(datap))
            m_temp, S_temp = Linearize(mean_pc, signalp, Mop, jop[i])
            d_current, R_current, N_current, gamma_current = compress(m_temp, S_temp,
                Mop, kmax, Ro=Rp, gamma=gamma, multifield=True, numpy=True,
                k_gamma = k_gamma)
            d_pc[i][:len(d_current)] = d_current
            R_pc[i][:len(R_current)] = R_current
            Npc_inv[i][:len(N_current)] = N_current
            gamma_min_pc[i] = gamma_current
        else: continue
    comptime_pc = time.perf_counter() - comptime_pc
    
    SteamRoller =  Reshaper(patched_signal.target,
            ift.UnstructuredDomain((patches_number,np.prod(patch_dim))))
    d_pc = ift.from_global_data(ift.UnstructuredDomain(d_pc.shape), d_pc)
    R_pc = Matmul(SteamRoller(patched_signal).target, d_pc.domain, R_pc) @ SteamRoller
    R_pc.adjoint(d_pc.to_global_data())
    ift.extra.consistency_check(R_pc)
    
    signal_responsepc = R_pc(patched_signal)
    
    
    likelihood_pc = ift.GaussianEnergy(mean=d_pc, inverse_covariance=ift.DiagonalOperator(ift.from_global_data(signal_responsepc.target, Npc_inv)))(signal_responsepc)
    Hpc = ift.StandardHamiltonian(likelihood_pc, ic_sampling)
    
    Rd0_pc = RGCP.adjoint_times(
                R_pc.adjoint_times(
                    d_pc)
            )
        
      
    #########################
    ###Patchwise Compression###
    #########################
    
    print("Patching Inference")
    
    for l in range(n_comp):
        inftime_pc.append(-time.perf_counter())
        for i in range(n_rep):
            # Draw new samples and minimize KL
            KL_pc = ift.MetricGaussianKL(mean_pc, Hpc, N_samples, mirror_samples=True)
            KL_pc, convergence = minimizer(KL_pc)
            mean_pc = KL_pc.position
            print((l*n_rep+i+1)*100/(n_comp*n_rep))
        inftime_pc[-1] += time.perf_counter()

        if l < n_comp-1:
            d_pc  = np.zeros((patches_number, kmax))
            R_pc  = np.zeros((patches_number, kmax, np.prod(patch_dim)))
            Npc_inv = np.zeros((patches_number, kmax))
            gamma_min_pc = [[0.,1.]]*patches_number
            
            for i in range(patches_number):
                print(100*i/patches_number)
                if i in isvalidpatch:
                    selector = ift.DomainTupleFieldInserter(patched_signal.target, 0, (i,)).adjoint
                    signalp = selector @ patched_signal
                    m_temp, S_temp = Linearize(mean_pc, signalp, Mop, jop[i])
                    d_current, R_current, N_current, gamma_current = compress(m_temp, S_temp,
                        Mop, kmax, Ro=Rp, gamma=gamma, multifield=True, numpy=True,
                        k_gamma = k_gamma)
                    d_pc[i][:len(d_current)] = d_current
                    R_pc[i][:len(R_current)] = R_current
                    Npc_inv[i][:len(N_current)] = N_current
                    gamma_min_pc[i] = gamma_current
                else: continue
            
            d_pc = ift.from_global_data(ift.UnstructuredDomain(d_pc.shape), d_pc)
            R_pc = Matmul(SteamRoller(patched_signal).target, d_pc.domain,R_pc) @ SteamRoller
            R_pc.adjoint(d_pc.to_global_data())
            
            signal_responsepc = R_pc(patched_signal)
            Npc_inv = ift.DiagonalOperator(ift.from_global_data(signal_responsepc.target, Npc_inv))
            
            likelihood_pc = ift.GaussianEnergy(mean=d_pc, inverse_covariance=Npc_inv)(signal_responsepc)
            Hpc = ift.StandardHamiltonian(likelihood_pc, ic_sampling)
    sumtime_pc = time.perf_counter() - sumtime_pc


    # Draw posterior samples
    KL_pc = ift.MetricGaussianKL(mean_pc, Hpc, N_samples)
    scpc = ift.StatCalculator()
    for sample in KL_pc.samples:
        scpc.add(signal(sample + KL_pc.position))
        
 
    Rd1_pc = RGCP.adjoint_times(
                    R_pc.adjoint_times(
                        d_pc)
                    )
    

    with open(path+'/rcounter.txt','a') as f:
        f.write("Patch: Rcounter = {}\n".format(CntR.Rcounter)
            )
    
    with open(path+"/NonLin_pc.pk",'wb') as f:
          pickle.dump([KL_pc, R_pc, Npc_inv, d_pc, scpc.mean, scpc.var, Rd0_pc,
              Rd1_pc, gamma_min_pc, inftime_pc, sumtime_pc], f)
    
    
    
    print("Time for usual, compression and patchcomp for n_rep inference steps:")
    print(inftime_o)
    print(inftime_c)
    print(inftime_pc)

    print("Totel time for usual, compression and patchcomp method and inference:")
    print(sumtime_o)
    print(sumtime_c)
    print(sumtime_pc)

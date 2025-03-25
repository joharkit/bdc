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

import pickle
import sys
import time

import nifty_gridder
import numpy as np
from scipy.sparse import vstack
from scipy.sparse.linalg import aslinearoperator

import matplotlib as mpl
mpl.use('Agg')
import nifty5 as ift
import resolve as rve
from bdc.Cutter import SpiderWebCutter, PatchCompCut, ChessboardCutter
from tools.Niftize import Niftize
from tools.reshaper import Reshaper, Counter
from bdc.tests import ResponseCutterTest

import matplotlib.pyplot as plt

#For creating the directories Plots and Wimmerl
import os

if __name__ == '__main__':
    np.random.seed(42)
    # compression parameters:
    datamode = "vis"
    # Number of compressed data points per patch
    kmax = 64
    # Compression factor
    gamma = 0.99
    # Number of compressions
    n_comp = 3
    # Number of KLMinimizations after each Compression
    n_rep = 3
    # Number of KLMinimizations for original inference
    n_o = 3 * n_rep * n_comp
    # number of samples used to estimate the KL
    N_samples = 4
    # and posterior
    N_posterior_samples = 10
    # if Chessboard==True ChessboardCutter, else SpiderWebCutter
    Chessboard=True

    #Set True, if you want to plot PatchedData.pdf
    PlotUVPatches=True

    # Choose if you want to do
    # 0 == Usual Compresion
    # 1 == Patch and Compress
    # 2 == Minimize Patchcomp
    if len(sys.argv) > 2:
        mode = int(sys.argv[2])
    else:
        mode = 1

    nthreads = 4
    ift.fft.set_nthreads(nthreads)
    nifty_gridder.set_nthreads(nthreads)

    try:
        os.makedirs("Plots", mode=0o777)
        print("created directory Plots")
    except:
        pass
    try:
        os.makedirs("Wimmerl", mode=0o777)
        print("created directory Wimmerl")
    except:
        pass

    dht = rve.DataHandler((256, 256),
                          rve.DEG2RAD*np.array((0.24, 0.24)),
                          200000,
                          shift=(0, 0.1*rve.DEG2RAD),
                          datamode=datamode)
    imgr = rve.Imager(20, dht)
    dct = {
        'a': 1,
        'k0': 1,
        'sm': -4,
        'sv': 0.5,
        'im': -1.9786410264310597,
        'iv': 2,
        'zm': 0.000287746133206174,
        'zmstd': 3.509192675942882e-05
    }
    d = dht.vis()
    uv = dht._uvw[:, 0:2]

    # sky --R1--> signal (on grid) --R2--> data
    # (data format visibilities or ms (measurement sets))
    # Piper replaces R2 by a big sparse response
    CntR = Counter((dht.R_degridder()).target)
    R2 = CntR @ dht.R_degridder()
    R1 = dht.R_rest()
    R = R2 @ R1
    N = ift.makeOp(dht.invvar()).inverse
    Ninv = ift.makeOp(dht.invvar())
    imgr.init_diffuse(**dct)
    sky = imgr.sky
    pspec = imgr._diffuse.pspec

    signal = R1 @ sky
    signal_response = R2(signal)

    # Minimization parameters
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradInfNormController(name='Newton', tol=1e-7, iteration_limit=10)
    minimizer = ift.NewtonCG(ic_newton)

    # Set up likelihood and information Hamiltonian
    likelihood = ift.GaussianEnergy(mean=d, inverse_covariance=Ninv)(signal_response)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    if mode == 0:
        filename = "Plots/bdc_Resolve_{}_".format(gamma)+"CompTo_{}_datapoints_".format(kmax)+"_{}_".format(N_samples)+"{}_".format(n_o)+"{}.pdf"
        CntR.Rcounter = 0
        inftime_o = []
        sumtime_o = -time.perf_counter()
        inftime_o.append(-time.perf_counter())
        for i in range(n_o):
            # Draw new samples and minimize KL
            KL = ift.MetricGaussianKL(mean, H, N_samples, mirror_samples=True)
            KL, convergence = minimizer(KL)
            mean = KL.position
            print((i+1)*100//n_o)

        inftime_o[-1] += time.perf_counter()
        sumtime_o += time.perf_counter()

        # Draw posterior samples
        KL = ift.MetricGaussianKL(mean, H, N_posterior_samples)
        sc = ift.StatCalculator()
        for sample in KL.samples:
            sc.add(sky(sample + KL.position))

        with open("Wimmerl/sc_o_{}.pk".format(n_o), 'wb') as f:
            pickle.dump([filename.format("o_{}"), KL.position, KL.samples, pspec, sc.mean, sc.var, CntR.Rcounter, inftime_o, sumtime_o], f)

    elif mode == 1:
        # In order to distinguish different reconstructions, they are
        # distinguished by the time, they were performed.
        # This way a run of several compressions can be split.
        if len(sys.argv) == 4:
            current_time = int(sys.argv[3])
        else:
            current_time = 0  # TEMP

        if current_time == 0:
            current_time = int(time.perf_counter())
            mean_bs = initial_mean
        else:
            with open("Wimmerl/{}-mean_bs.pk".format(current_time), 'rb') as f:
                mean_bs = pickle.load(f) 

        CntR.Rcounter = 0
        inftime_pc = []
        sumtime_pc = -time.perf_counter()
        patchnumber = [64, 64]
        
        # The uv-coordinates are assigned to the different Patches.
        # ChessboardCutter would structure the patches like a chessboard
        # SpiderWebCutter in equiradial and equiangular patches
        print(uv.shape)
        if Chessboard:
            Cuttr = ChessboardCutter
        else:
            Cuttr = SpiderWebCutter
        boxuindex, boxvindex, box_index, index, minvec, maxvec = Cuttr(uv, patchnumber=patchnumber, plotStatistics=True)

        for klajsdf in range(n_comp):
            current_time += 1
            # PatchCompCut first assigns the visibilities (d=dht.vis()) and their variances (dht.var()) according to the mapping of the uv-coordinates to the different patches
            # Then it Compresses the data for each patch seperately
            # And sticks them alltogether in the end
            # It also makes a backup (pickle) of the single compresssed responses, data and noise covariances
            bsdc, BigSparseResponse, bsNc_inv, pickletime, gamma_min = PatchCompCut(
                mean_bs, signal, R2, Ninv, d, kmax, R1.target.shape, patchnumber, uv, boxuindex, boxvindex, box_index, index, current_time, dht, datamode=datamode, gamma=gamma)
            sumtime_pc -= pickletime
            signal_bsresponse = BigSparseResponse(signal)
            likelihood_bs = ift.GaussianEnergy(mean=bsdc, inverse_covariance=bsNc_inv)(signal_bsresponse)
            H_bs = ift.StandardHamiltonian(likelihood_bs, ic_sampling)
            filename = "Plots/{}_".format(current_time)
            filename += "bdc_Resolve_CompTo_insg{}".format(bsdc.size)+"_max{}pP_datapoints_".format(kmax)+"_{}_".format(N_samples)+"{}_".format(n_rep)+"{}_".format(n_comp)
            if Chessboard:
                filename+="Chessboard_"
            else:
                filename+="SpiderWeb_"
            filename+="{}.pdf"


            # patchcompress
            inftime_pc.append(-time.perf_counter())
            for i in range(n_rep):
                # Draw new samples and minimize KL
                KL_bs = ift.MetricGaussianKL(mean_bs, H_bs, N_samples, mirror_samples=True)
                KL_bs, convergence_bs = minimizer(KL_bs)
                mean_bs = KL_bs.position
                print((klajsdf*n_rep+i+1)*100/(n_comp*n_rep))
            inftime_pc[-1] += time.perf_counter()
            sumtime_pc+= time.perf_counter()
            with open("Wimmerl/{}-mean_bs.pk".format(current_time), 'wb') as f:
                pickle.dump(mean_bs, f)
            with open("Wimmerl/{}-params.txt".format(current_time), 'w') as f:
                f.writelines(["datamode = {}\n".format(datamode),
                    "kmax = {}\n".format(kmax),
                    "gamma = {}\n".format(gamma),
                    "n_comp = {}\n".format(n_comp),
                    "n_rep = {}\n".format(n_rep),
                    "N_samples = {}\n".format(N_samples),
                    "N_posterior_samples = {}\n".format(N_posterior_samples),
                    "Chessboard = {}\n".format(Chessboard),
                    "k_c = {}".format(bsdc.size),
                    ])
            sumtime_pc-=time.perf_counter()

        sumtime_pc += time.perf_counter()
        with open("Wimmerl/{}-times2.pk".format(current_time), 'wb') as f:
            pickle.dump([CntR.Rcounter, inftime_pc,sumtime_pc, gamma_min], f)
        # Draw posterior samples
        KL_bs = ift.MetricGaussianKL(mean_bs, H_bs, N_posterior_samples)
        sc_bs = ift.StatCalculator()
        for sample in KL_bs.samples:
            sc_bs.add(sky(sample + KL_bs.position))

    elif mode == 3:
        if len(sys.argv) == 4:
            current_time = int(sys.argv[3])
        else:
            current_time = int(input("Which mean do you want to plot? (integer only)"))

        filename = "Plots/{}_".format(current_time)
        filename += "bdc_Resolve_CompTo_{}_datapoints_".format(kmax)+"_{}_".format(N_samples)+"{}_".format(n_rep)+"{}_".format(n_comp)+"{}.pdf"
        
        
        with open("Wimmerl/{}-mean_bs.pk".format(current_time), 'rb') as f:
            mean_bs = pickle.load(f) 

        with open("Wimmerl/{}-npdc_current.pk".format(current_time), 'rb') as f:
            npdc_current = pickle.load(f)
        with open("Wimmerl/{}-npNc_inv_current.pk".format(current_time), 'rb') as f:
            npNc_inv_current = pickle.load(f)
        with open("Wimmerl/{}-BigSparseResponse.pk".format(current_time), 'rb') as f:
            BigSparseResponse = pickle.load(f)
        
        # Build dc, Rc, Nc
        npdc = np.concatenate(npdc_current)
        npNinv_diag = np.concatenate(npNc_inv_current)
        BigSparseResponse = aslinearoperator(vstack(BigSparseResponse))
        kmax = npdc.size
        BigSparseResponse = Niftize(ift.RGSpace([BigSparseResponse.shape[1]]), ift.UnstructuredDomain((npdc.size)),
                                    BigSparseResponse)
        BigSparseResponse = BigSparseResponse @ Reshaper(domain=signal.target, target=BigSparseResponse.domain)
        signal_bsresponse = BigSparseResponse(signal)

        bsdc = ift.from_global_data(BigSparseResponse.target, npdc)
        bsNc_inv = ift.DiagonalOperator(ift.from_global_data(BigSparseResponse.target, npNinv_diag))
        likelihood_bs = ift.GaussianEnergy(mean=bsdc, inverse_covariance=bsNc_inv)(signal_bsresponse)
        H_bs = ift.StandardHamiltonian(likelihood_bs, ic_sampling)


        filename = "Plots/{}_".format(current_time)
        filename += "bdc_Resolve_CompTo_insg{}".format(bsdc.size)+"_max{}pP_datapoints_".format(kmax)+"_{}_".format(N_samples)+"{}_".format(n_rep)+"{}_".format(n_comp)
        if Chessboard:
            filename+="Chessboard_"
        else:
            filename+="SpiderWeb_"
        filename+="{}.pdf"

        # Draw posterior samples
        KL_bs = ift.MetricGaussianKL(mean_bs, H_bs, N_posterior_samples)
        sc_bs = ift.StatCalculator()
        for sample in KL_bs.samples:
           sc_bs.add(sky(sample + KL_bs.position))

        filename_res = filename.format("BigSparseResults")
        with open("Wimmerl/{}-sc_bs.pk".format(current_time), 'wb') as f:
            pickle.dump([filename_res, KL_bs.position, KL_bs.samples, sc_bs.mean, sc_bs.var], f)


    elif mode == 2:
        current_time = int(input("Which data do you want to infere? (first number only)"))

        filename = "Plots/{}_".format(current_time)
        filename += "bdc_Resolve_CompTo_{}_datapoints_".format(kmax)+"_{}_".format(N_samples)+"{}_".format(n_rep)+"{}_".format(n_comp)+"{}.pdf"

        with open("Wimmerl/{}-npdc_current.pk".format(current_time), 'rb') as f:
            npdc_current = pickle.load(f)
        with open("Wimmerl/{}-npNc_inv_current.pk".format(current_time), 'rb') as f:
            npNc_inv_current = pickle.load(f)
        with open("Wimmerl/{}-BigSparseResponse.pk".format(current_time), 'rb') as f:
            BigSparseResponse = pickle.load(f)

        # Build dc, Rc, Nc
        npdc = np.concatenate(npdc_current)
        npNinv_diag = np.concatenate(npNc_inv_current)
        BigSparseResponse = aslinearoperator(vstack(BigSparseResponse))
        kmax = npdc.size
        BigSparseResponse = Niftize(ift.RGSpace([BigSparseResponse.shape[1]]), ift.UnstructuredDomain((npdc.size)),
                                    BigSparseResponse)
        BigSparseResponse = BigSparseResponse @ Reshaper(domain=signal.target, target=BigSparseResponse.domain)
        signal_bsresponse = BigSparseResponse(signal)

        bsdc = ift.from_global_data(BigSparseResponse.target, npdc)
        bsNc_inv = ift.DiagonalOperator(ift.from_global_data(BigSparseResponse.target, npNinv_diag))
        likelihood_bs = ift.GaussianEnergy(mean=bsdc, inverse_covariance=bsNc_inv)(signal_bsresponse)
        H_bs = ift.StandardHamiltonian(likelihood_bs, ic_sampling)

        mean_bs = mean
        # Patchcompress
        for i in range(n_rep):
            KL_bs = ift.MetricGaussianKL(mean_bs, H_bs, N_samples, mirror_samples=True)
            KL_bs, convergence_bs = minimizer(KL_bs)
            mean_bs = KL_bs.position
            print((i+1)*100/(n_rep))
        
        # Draw posterior samples
        KL_bs = ift.MetricGaussianKL(mean_bs, H_bs, N_posterior_samples)
        sc_bs = ift.StatCalculator()
        for sample in KL_bs.samples:
            sc_bs.add(sky(sample + KL_bs.position))

        filename_res = filename.format("BigSparseResults")
        plot = ift.Plot()
        plot.add(sc_bs.mean, title="Posterior Mean")
        plot.add(ift.sqrt(sc_bs.var), title="Posterior Standard Deviation")
        plot.output(ny=1, nx=2, xsize=12, ysize=6, name=filename_res)
        print("Saved results as '{}'.".format(filename_res))

        with open("Wimmerl/{}-mean_bs.pk".format(current_time), 'wb') as f:
            pickle.dump(mean_bs, f)

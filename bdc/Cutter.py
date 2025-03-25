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
# Author: Johannes Harth-Kitzerow

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import aslinearoperator

import nifty5 as ift
from bdc.Compress import compress, Linearize
from tools.DataShorter import DataShorter
from tools.Niftize import Niftize
from tools.reshaper import Reshaper


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho,phi):
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    return np.array([x,y])


def ResponseCutter(dht, data_mask):
    R2, Ninv, d = dht.select_from_datamask(data_mask)
    Mo_current = ift.SandwichOperator.make(R2, Ninv)
    jo_current = R2.adjoint_times(Ninv(d))
    return R2, Ninv, d, Mo_current, jo_current


def PatchCompCut(mean_bs, signal, R2, Ninv, d, kmax, domainsize, patchnumber, uv,
                 boxuindex, boxvindex, box_index, index, current_time, dht,
                 selection=None,
                 datamode="ms", Hammer=False,
                 gamma = 0.7):
    patches_number = np.prod(patchnumber)

    npdc_current = [[]]*patches_number
    BigSparseResponse = [[]]*patches_number
    npNc_inv_current = [[]]*patches_number
    comptime = []
    patchingtime = []
    setuptime=[]
    pickletime = 0
    gamma_min = [[]]*patches_number

    for l in range(patches_number):
        BigSparseResponse[l] = coo_matrix((0, np.prod(domainsize)))

    for l in range(patches_number):
        if l % (patches_number//100) == 0:
            print((100*l)//patches_number)

        # create array, with uv coordinate of all data (second and third coordinate) inside patch (first index)
        this_box_uv = np.copy(uv[(boxuindex[l] == index[:, 0]) & (boxvindex[l] == index[:, 1])])

        if len(this_box_uv) < 1:
            BigSparseResponse[l] = coo_matrix((0, np.prod(domainsize)))
            continue

        # See, if there are actually data in this patch and go to next patch if there are none
        number_of_data_inside_current_patch = len(this_box_uv[:, 0])
        setuptime.append(-time.perf_counter())

        # Get Response for this patch
        if Hammer:
            R2_current, Ninv_current, data_current_patch, Mo_current, jo_current = HammerResponseCutter(
                R2, Ninv, d, box_index, l)
        else:
            data_mask = box_index == l
            R2_current, Ninv_current, data_current_patch, Mo_current, jo_current = ResponseCutter(
                dht, data_mask)

        setuptime[-1] += time.perf_counter()
        comptime.append(-time.perf_counter())
        #1. Compress patch number l
        m_temp, S_temp = Linearize(mean_bs, signal, Mo_current, jo_current)
        npdc_current[l], npRc_current, npNc_inv_current[l], gamma_min[l] = compress(
            m_temp, S_temp, Mo_current,
            np.minimum((number_of_data_inside_current_patch), kmax),
            gamma=gamma, multifield=True, numpy=True)
        comptime[-1] += time.perf_counter()
        patchingtime.append(-time.perf_counter())
        # 2. Neglect small entries to improve sparcity
        npRc_current[abs(npRc_current.real) < 1e-8] = 0.
        spRc_current = coo_matrix((len(npdc_current[l]), np.prod(domainsize)))

        for k in range(len(npdc_current[l])):
            helplist2 = np.nonzero(npRc_current[k])[0]
            helplist1 = np.zeros(len(helplist2), dtype=np.int64) + k
            helplist0 = npRc_current[helplist1, helplist2]
            smat = coo_matrix((helplist0, (helplist1, helplist2)), shape=(len(npdc_current[l]), np.prod(domainsize)))
            spRc_current += smat
        # concatenate current patch's Response, data and Noise to whole matrices
        BigSparseResponse[l] = spRc_current
        spRc_current = aslinearoperator(spRc_current)
        patchingtime[-1]+=time.perf_counter()
        pickletime -= time.perf_counter()
        with open("Wimmerl/{}-BigSparseResponse.pk".format(current_time),
                  'wb') as f:
            pickle.dump(BigSparseResponse, f)
        with open("Wimmerl/{}-npdc_current.pk".format(current_time),
                  'wb') as f:
            pickle.dump(npdc_current, f)
        with open("Wimmerl/{}-npNc_inv_current.pk".format(current_time),
                  'wb') as f:
            pickle.dump(npNc_inv_current, f)
        with open("Wimmerl/{}-times.pk".format(current_time),
                  'wb') as f:
            pickle.dump([setuptime, patchingtime, comptime], f)
        pickletime += time.perf_counter()
    print(current_time)

    npdc = np.concatenate(npdc_current)
    npNinv_diag = np.concatenate(npNc_inv_current)

    BigSparseResponse = aslinearoperator(vstack(BigSparseResponse))
    BigSparseResponse = Niftize(ift.RGSpace([BigSparseResponse.shape[1]]), ift.UnstructuredDomain((npdc.size)), BigSparseResponse)
    BigSparseResponse = BigSparseResponse @ Reshaper(domain=signal.target, target=BigSparseResponse.domain)

    bsdc = ift.from_global_data(BigSparseResponse.target, npdc)
    bsNc_inv = ift.DiagonalOperator(
        ift.from_global_data(BigSparseResponse.target, npNinv_diag))
    return bsdc, BigSparseResponse, bsNc_inv, pickletime, gamma_min


def SpiderWebCutter(uv, patchnumber=[64,64],plotStatistics=True):
    boxuindex, boxvindex, box_index, index, minvec, maxvec = ChessboardCutter(np.transpose(np.array(cart2pol(uv[:, 0], uv[:, 1])), (1, 0)), patchnumber, plotStatistics=False, Polar=True)
    if plotStatistics:
         PlotPolarPatches(uv, minvec[0], maxvec[0], minvec[1], maxvec[1], box_index, patchnumber=patchnumber, color=False)
    return boxuindex, boxvindex, box_index, index, minvec, maxvec

def ChessboardCutter(uv, patchnumber=[64, 64], plotStatistics=True, Polar=False):
    patches_number = np.prod(patchnumber)

    # 1. ACTUAL PATCHING
    minu = np.amin(uv[:, 0])
    maxu = np.amax(uv[:, 0])
    minv = np.amin(uv[:, 1])
    maxv = np.amax(uv[:, 1])
    # Create 64x64=64^2 boxes lying in [minu,maxu] and [minv,maxv]
    minvec = np.array([minu, minv])
    maxvec = np.array([maxu, maxv])
    # index: array, datapoint[i] lies in box number index[i](= [index_u, index_v]
    index = np.floor((uv - minvec)/(maxvec - minvec)*(0.99999)* np.array(patchnumber)).astype(np.int64)
    # box_index is index as absolute number counting the u-coordinate first and then v , i.e.
    box_index = index[:, 1]*patchnumber[0] + index[:, 0]

    helpindex = np.arange(patches_number)
    boxuindex = helpindex % patchnumber[0]
    boxvindex = helpindex//patchnumber[0]
    if plotStatistics:
        PlotPatchStatistics(box_index, uv, patchnumber=patchnumber)
        if Polar:
            PlotPolarPatches(uv, minu, maxu, minv, maxv, box_index, patchnumber=patchnumber)
        else:
            PlotChessPatches(uv, minu, maxu, minv, maxv, patchnumber=patchnumber)
    return boxuindex, boxvindex, box_index, index, minvec, maxvec


def HammerResponseCutter(R2, Ninv, d, box_index, patch_number):
    bindices = np.where(box_index == patch_number)[0]
    mask_current = np.array([(box_index == patch_number)])[0]
    domain = ift.UnstructuredDomain(bindices.size)
    domain2 = ift.UnstructuredDomain(np.nonzero(mask_current)[0].size)
    data_current_patch = ift.from_global_data(domain, d.to_global_data()[bindices])

    DS = DataShorter(R2.target, domain2, mask_current)
    R2_current = DS @ R2
    Ninv_current = DS @ Ninv @ DS.adjoint

    Mo_current = ift.SandwichOperator.make(R2_current, Ninv_current)
    jo_current = R2_current.adjoint_times(DS(Ninv(d)))
    return R2_current, Ninv_current, data_current_patch, Mo_current, jo_current


def PlotPatchStatistics(box_index, uv, patchnumber=[64, 64]):
    patches_number = np.prod(patchnumber)
    # 1a) PLOTTING HOW MANY DATA POINTS ARE IN HOW MANY PATCHES
    plt.ylabel('Number of data inside patch')
    plt.xlabel('patch number')
    plt.plot(np.bincount(box_index, minlength=patches_number))
    plt.savefig('Plots/NumberDataVsPatch.pdf')
    plt.close()
    plt.xlabel('number of data')
    plt.ylabel('number of patches with certain amount of data')
    logbins = np.logspace(np.log10(0.1), np.log10(np.amax(box_index) + 0.1),
                          100)
    plt.hist(np.bincount(box_index) + 0.1, bins=logbins, log=True)
    plt.xscale('log')
    plt.savefig('Plots/NumberPatchesVsNumberDataHist.pdf')
    plt.close()
    plt.xlabel('number of data')
    plt.ylabel('number of patches with number of data')
    plt.plot(np.bincount(np.bincount(box_index)))
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('Plots/NumberPatchesVsNumberData.pdf')
    plt.close()
    print("Plotted PatchStatistics in subfolder /Plots/")


def PlotChessPatches(uv, minu, maxu, minv, maxv, patchnumber=[64, 64]):
    plt.plot(uv[:, 0], uv[:, 1], ',')
    for asdf in range(patchnumber[0] + 1):
        for fdsa in range(patchnumber[1] + 1):
            plt.plot([minu + np.float64(asdf)*(maxu - minu)/patchnumber[0], minu + np.float64(asdf)*(maxu - minu)/patchnumber[0]],
                     [minv, maxv], 'k', linewidth=0.3)
            plt.plot([minu, maxu], [minv + np.float64(fdsa)*(maxv - minv)/patchnumber[1], minv + np.float64(fdsa)*(maxv - minv)/patchnumber[1]],
                     'k', linewidth=0.3)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.xlim(minu, maxu)
    plt.ylim(minv, maxv)
    plt.savefig('Plots/PatchedData.pdf')
    print("Plotted PatchedData in subfolder /Plots/")

def PlotPolarPatches(uv, minr, maxr, minphi, maxphi, box_index, patchnumber=[64, 64], color=False):
    fig, ax = plt.subplots()
    if color:
        for ii in np.arange(np.prod(patchnumber)):
            uv_current = uv[ box_index == ii ]
            plt.plot(uv_current[:,0],uv_current[:,1],",", zorder=3)
    else:
        plt.plot(uv[:, 0], uv[:, 1], ',')
    for fdsa in range(patchnumber[1] + 1):
        #Draw equiangular lines
        
        plt.plot([0, maxr*np.cos(minphi + np.float64(fdsa)*(maxphi - minphi)/patchnumber[1])], [0, maxr*np.sin(minphi + np.float64(fdsa)*(maxphi - minphi)/patchnumber[1])],'k', linewidth=0.2)
    for asdf in range(patchnumber[0] + 1):
        #Draw equiradial lines
        ax.add_artist(plt.Circle((0,0),minr+np.float64(asdf)*(maxr-minr)/patchnumber[0],fill=False, clip_on=True, linewidth=0.2))
    ax.set_zorder(2)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.xlim(-maxr,maxr)
    plt.ylim(0,maxr)
    plt.savefig('Plots/PatchedData.pdf')
    print("Plotted PatchedData in subfolder /Plots/")

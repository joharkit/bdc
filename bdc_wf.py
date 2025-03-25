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
import matplotlib as mpl
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, eigsh, LinearOperator
from tools.dense_operator import DenseOperator
from bdc.Compress import compress
from tools.pca import bayesian_pca as bpca
from sklearn.decomposition import PCA as sklpca

from scipy.optimize import curve_fit

import sys
# sys.argv 0 for easy setup, 1 for masked
if len(sys.argv) == 2:
    easy = 1 - int(sys.argv[1])
else:
    easy = 0

try:
    os.makedirs("../Plots", mode=0o777)
    print("created directory Plots")
except:
    pass

filename = "./Plots/bdc_wf_{}.pdf"
if easy:
    filename=filename.format("easy_{}")

np.random.seed(41)
# Set the dimension of the mock data space
if easy:
    Ndim = 2048
else:
    Ndim = 256

gamma = .99 
kmax = 4
show = False
BaPCA = 1-easy

def explicify(Op):
    matrix_expl = np.zeros((Op.target.size, Op.domain.size))
    for ii in range(Ndim):
        x = np.zeros(Ndim)
        x[ii] = 1.
        x = ift.from_global_data(sp, x)
        matrix_expl[ii] = Op(x).to_global_data()
    return matrix_expl

def explicit_unc(D_inv):
    return np.sqrt(np.diag(np.linalg.inv(explicify(D_inv))))

#########
# Setup #
#########

# specify domain of the signal
sp = ift.RGSpace(Ndim)
# Create the harmonic space of the real grid
hsp = sp.get_default_codomain()
# init Hartley Transform
ht = ift.HartleyOperator(hsp, target=sp)

# init power spectrum
def amplitude(k):
    return np.sqrt(2e4/(1.+(k/2e-1)**4))

###GETTING THE PRIOR COVARIANCE OUT OF THE POWER SPECTRUM###

# Create PowerSpace
psp = ift.PowerSpace(hsp)
# Distribute points from power spectrum into (higher dimensional) harmonic space
PD = ift.PowerDistributor(hsp, psp)
S_diag = PD(ift.PS_field(psp, amplitude)**2)
Sh = ift.DiagonalOperator(S_diag)
S = ift.SandwichOperator.make(ht.inverse, Sh)

### CREATE SOME MASK ###
mask = np.zeros(sp.shape)
if easy:
    data_region = np.arange(896, Ndim-896)
else:
    data_region = np.append(np.arange(35,45), np.arange(60,90)) # 
mask[data_region] = 1

### CREATE SOME NOISE ###
noise_diag = np.ones(Ndim)
if easy:
    noise_diag*=.2e-2
else:
    # Set values of the noise diagonal for the first 180 entries to 0.002
    noise_diag[:180] = .2e-2
    # Set values of the noise diagonal for the all entries after 80 to 0.004
    noise_diag[80:] = .4e-2#.2e-2
# init noise covariance
N = ift.DiagonalOperator(ift.from_global_data(sp, noise_diag**2))
# Get the response for the mask and the real grid
R = ift.DiagonalOperator(ift.from_global_data(sp,mask))

# Draw mock signal
MOCK_SIGNAL = S.draw_sample()
# Draw mock noise
MOCK_NOISE = N.draw_sample()

data = R(MOCK_SIGNAL) + MOCK_NOISE


#################
# Wiener Filter #
#################

j = R.adjoint_times(N.inverse_times(data))
Mo = ift.SandwichOperator.make(R,N.inverse)
MoS = Mo(S)

IC = ift.GradientNormController(iteration_limit = 500, tol_abs_gradnorm = 1e-3)
D_inv = ift.WienerFilterCurvature(R, N, S, IC, IC)
m = D_inv.inverse(j)

npm = m.to_global_data()
# Plot MOCK_SIGNAL
plt.plot(np.arange(Ndim), MOCK_SIGNAL.to_global_data(), color='g')
# Plot data as dots with errorbars
plt.errorbar(np.arange(Ndim)[data_region], data.to_global_data()[data_region], fmt = ',', yerr = noise_diag[data_region], color='.5', linewidth=0.1)
plt.xlabel("position")
plt.ylabel("signal")
plt.yticks([-1.7e-1, -1.6e-1, -1.5e-1, -1.4e-1, -1.3e-1] if not easy else None)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2),useMathText=True)
plt.savefig(filename.format("Setup"))
plt.close()


###############
# Compression #
###############

def numpyze(Op):
    # translate NIFTy oject to numpy
    in_shape = Op.domain.shape
    return(lambda x: Op(ift.from_global_data(Op.domain, x.reshape(in_shape))).to_global_data().reshape(-1))

# Make Op a linear operator
#! No Multifield!
def LOp(Op):
    dim_in = np.prod(Op.domain.shape)
    dim_out = np.prod(Op.target.shape)
    return LinearOperator(dtype=np.float64, shape=(dim_out, dim_in), matvec=numpyze(Op))

dc, r, musqd, gamma_min = compress(m, S, Mo, kmax, gamma=gamma, numpy=True,
        k_gamma=np.int(np.sum(mask))-1)

print("gamma_min = "+str(gamma_min))
print("k_c = "+str(dc.size))

### CALCULATING FOR MS ##
r = r.T
deltasqd = 1./(musqd+1.)
plt.plot(-np.log(np.sqrt(deltasqd)), 'o')
plt.xlabel("position")
plt.ylabel("signal")
plt.savefig(filename.format("InformationGain"))

#######
# PCA #
#######

if BaPCA:
    pca_trafo, N_pca_inv = bpca(R, S, N, kmax = kmax, numpy=False)
    d_pca = pca_trafo(data)
    R_pca = pca_trafo @ R
    j_pca = R_pca.adjoint_times(N_pca_inv(d_pca))
    IC_pca = ift.GradientNormController(iteration_limit = d_pca.domain.size)
    D_pca_inv = ift.WienerFilterCurvature(R_pca, N_pca_inv.inverse, S, IC_pca,
            IC_pca)
    m_pca = D_pca_inv.inverse(j_pca)

    unc_pca = explicit_unc(D_pca_inv)

    pca_trafo, _ = bpca(R, S, N, kmax = kmax, numpy=True)


'''
Plot eigenvectors of BDC
if easy:
    also fit and plot Chabyshev polynomials
else:
    also fit eigenvectors of BaPCA
'''
if easy:
    def T0(x,ampl,stretch,shift):
        x = stretch*x+shift
        T = 0.*x + 1.
        return ampl*T

    def T1(x,ampl,stretch,shift):
        x = stretch*x+shift
        T = x
        return ampl*T

    def T2(x,ampl,stretch,shift,y):
        x = stretch*x+shift
        T = 2.*x**2 - 1
        return ampl*T+y

    def T3(x,ampl,stretch,shift,y):
        x = stretch*x+shift
        T = 2.*x*(2.*x**2 - 1) - x
        return ampl*T

    popt, pcov = curve_fit(T0,data_region, np.sqrt(musqd[0])*r[data_region,0])
    plt.plot(np.arange(Ndim), np.sqrt(musqd[0])*r[:,0], label = 'EV0', color=(1,0,0,1))
    plt.plot(data_region, T0(data_region, *popt), dashes=[2,2], color=(0.8,0.1,0.1,1), label = 'Cheb0')

    popt, pcov = curve_fit(T1,data_region, np.sqrt(musqd[1])*r[data_region,1])
    plt.plot(np.arange(Ndim), np.sqrt(musqd[1])*r[:,1], label = 'EV1', color='orange')
    plt.plot(data_region, T1(data_region, *popt), dashes=[2,2], label = 'Cheb1', color='orange')

    popt, pcov = curve_fit(T2,data_region, np.sqrt(musqd[2])*r[data_region,2])
    plt.plot(np.arange(Ndim), np.sqrt(musqd[2])*r[:,2], label = 'EV2', color='g')
    plt.plot(data_region, T2(data_region, *popt), dashes=[2,2], label = 'Cheb2', color='g')

    popt, pcov = curve_fit(T3,data_region, np.sqrt(musqd[3])*r[data_region,3])
    plt.plot(np.arange(Ndim), np.sqrt(musqd[3])*r[:,3], label = 'EV3', color='b')
    plt.plot(data_region, T3(data_region, *popt), dashes=[2,2], label = 'Cheb3', color='b')

    plt.gca().set_xlim([0.425*Ndim, 0.575*Ndim])
else:
    cls = ['r','orange','g','b']
    sgn = [1.,1.,-1.,-1.]
    for ii in range(4):
        plt.plot(np.arange(Ndim), np.sqrt(musqd[ii])*r[:,ii], label =
                'EV{}'.format(ii), color=cls[ii])
        if BaPCA:
            ampltd = sgn[ii]*np.linalg.norm(np.sqrt(musqd[ii])*r[:,ii])/np.linalg.norm(pca_trafo[ii])
            plt.plot(np.arange(Ndim), ampltd*pca_trafo[ii],
                label ='PCA-EV{}'.format(ii), dashes=[2,2], color=cls[ii])
        # Plot noise threshold
    plt.plot([79.5,79.5],[-150.,150.],':', color='grey')
    plt.xlim([30, 100])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2),useMathText=True)
plt.xlabel("position")
plt.ylabel("eigenvector")
plt.savefig(filename.format("EigVecs"))
plt.close()


Rc = DenseOperator(sp, r.T)
Nc_diag = ift.from_global_data(Rc.target, musqd)
Nc_inv = ift.DiagonalOperator(Nc_diag)

RcSRc = ift.SandwichOperator.make(Rc.adjoint, S)
RcSRc_inv = ift.InversionEnabler(RcSRc, IC).inverse
dc = (RcSRc + Nc_inv.inverse) (RcSRc_inv(Rc(m)))



############################
# Compressed Wiener Filter #
############################

jc = Rc.adjoint_times(Nc_inv(dc))
Dc_inv = ift.WienerFilterCurvature(Rc, Nc_inv.inverse, S, IC, IC)
mc = Dc_inv.inverse(jc)

unc1 = explicit_unc(D_inv)
unc2 = explicit_unc(Dc_inv)

############
# Plotting #
############

unccolors = {
        'o'     :   'tab:purple',
        'c'     :   'tab:red',
        'pca'   :   'tab:cyan'
        }

meancolors = {
        'o'     :   'tab:purple',
        'c'     :   'tab:red',
        'pca'   :   'tab:cyan'
        }


print("Plotting")
npmc = mc.to_global_data()
for ii in range(2):
    plt.plot(np.arange(Ndim), npm, '--', label='WF', color=meancolors['o'])
    plt.plot(np.arange(Ndim), npmc, '-.', label='compressed WF',
    color=meancolors['c'])
    if BaPCA:
        plt.plot(np.arange(Ndim), m_pca.to_global_data(), label='PCA WF',
                color=meancolors['pca'],dashes=[2,2],)
    plt.plot(np.arange(Ndim), MOCK_SIGNAL.to_global_data(), label='Ground Truth', color='g')
    plt.fill_between(np.arange(Ndim), npmc+unc2, npmc-unc2, alpha=0.15,
            edgecolor = 'none', facecolor = unccolors['c'])# label='uncertainty compressed'
    plt.fill_between(np.arange(Ndim), npm+unc1, npm-unc1, alpha=0.15,
            edgecolor='none', facecolor=unccolors['o']) # label='original uncertainty', color=(0,0,.5,.5))
    if BaPCA:
        plt.fill_between(np.arange(Ndim),
                m_pca.to_global_data()+unc_pca,
                m_pca.to_global_data()-unc_pca,
                alpha=0.15,
                edgecolor='none',
                facecolor=unccolors['pca']) # label='pca uncertainty', color=(0,0,.5,.5))
    plt.fill_between(np.arange(Ndim), npm+unc1, npm-unc1,
            edgecolor='w', facecolor='none', hatch='|', linewidth=0.0) # label='original uncertainty', color=(0,0,.5,.5))
    plt.fill_between(np.arange(Ndim), npmc+unc2, npmc-unc2,
            edgecolor = 'w', facecolor = 'none', hatch = '//', linewidth=0.0)# label='uncertainty compressed'
    if BaPCA:
        plt.fill_between(np.arange(Ndim),
                m_pca.to_global_data()+unc_pca,
                m_pca.to_global_data()-unc_pca,
                edgecolor='w', facecolor='none', hatch = '-', linewidth=0.0)# label='pca uncertainty', color=(0,0,.5,.5))
    if ii == 0:
        plt.xlabel("position")
        plt.ylabel("mean")
        plt.yticks([-3.e-2, -2.5e-2, -2.e-2, -1.5e-2] if easy else None)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2),useMathText=True)
        plt.savefig(filename.format("Reconstruction"))
        plt.close()
    else:
        if easy:
            plt.xlim((0.4*Ndim, 0.6*Ndim))
            plt.ylim((-0.0250, -0.0175))
        else:
            plt.xlim((0.1*Ndim,0.35*Ndim))
            plt.ylim((-0.165,-0.14))
        plt.xlabel("position")
        plt.ylabel("mean")
        plt.yticks([-.17,-.16,-.15,-.14] if not easy else None)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2), useMathText=True)
        plt.savefig(filename.format("Reconstruction_Zoom"))
        plt.close()


plt.plot(np.arange(Ndim), 2.*(unc2-unc1)/(unc2+unc1),
        label='comp-orig uncertainty',
        color = 'tab:red')
if BaPCA:
    plt.plot(np.arange(Ndim), 2.*(unc_pca-unc1)/(unc_pca+unc1),dashes=[2,2],
            label='pca-orig uncertainty',
            color = 'tab:cyan')
plt.xlabel("position")
plt.ylabel("rel. uncertainty")
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2),useMathText=True)
plt.savefig(filename.format("Unc"))
plt.close()

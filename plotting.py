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

import sys
import pickle
import numpy as np
import nifty5 as ift
import resolve as rve
import matplotlib as mpl
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.sparse.linalg import eigs, eigsh, LinearOperator
from tools.dense_operator import DenseOperator
from tools.mf_remover import MFRemover
from tools.reshaper import Reshaper, Transposer, Matmul
from tools.matmul import StaticMatMul
from resolve.constants import SPEEDOFLIGHT, ARCMIN2RAD
from bdc_nl import make_checkerboard_mask

import time

from bdc.Compress import compress
from bdc_nl import RGCheckerboardPatcher

np.random.seed(42)

IC=ift.GradientNormController(iteration_limit = 500, tol_abs_gradnorm =1e-3)
N_samples = 20

# parameters for nl
nl_Wimmerl = "Wimmerl/1607078074/"

# parameters for rslv
n_comp=3
current_time = 1599848426
rs_Wimmerl = "Wimmerl/rslv_{}/".format(current_time)

cl = {
        'o'   :   'tab:purple',
        'c'  :   'tab:red', 
        'pc' :   'tab:blue',
        'pca' :    'tab:cyan',
        }
dsh = {
        'o'  :   '--',
        'c'  :   '-.',
        'pc'  :   ':',
        'pca'  :   ':',
        }
htch = {
        'o' :   '|',
        'c' :   '//',
        'pc' :   '-',
        'pca' :   '-'
        }

width=542.02501#pt
figwidth = width/72.27
figheight = figwidth*0.75

def getPlotSpecifications(inftype):
    '''
    Returns the colors, dashes and hatches depending on the type of plot

    Parameters
    ----------
    inftype     :   array of inference types plotted ('o','c','pc','pca')
    '''
    global cl
    global dsh
    global htch
    colors = [cl[ii] for ii in inftype]
    dashes = [dsh[ii] for ii in inftype]
    hatches = [htch[ii] for ii in inftype]
    return colors, dashes, hatches

def RoundDigit(x,digits=1):
    pot = int(np.ceil(np.log10(x)))
    x = int(np.ceil(x*10**(digits-pot)))
    x = x*10**(pot-digits)
    return x

def plotting_init():
    nice_fonts = {
        'text.usetex': True,
        'font.family': 'serif',
        'axes.labelsize': 6,
        'font.size': 6,
        'legend.fontsize': 6,
        'xtick.labelsize': 6,
        'xtick.labeltop': True,
        'xtick.top': True,
        'xtick.labelbottom': False,
        'xtick.bottom': False,
        'ytick.labelsize': 6,
    }
    nice_fonts['text.latex.preamble'] = [
        r'\usepackage{mathptmx}', r'\usepackage{nicefrac}'
    ]
    mpl.rcParams.update(nice_fonts)
    settings = {'image.cmap': 'afmhot', 'image.origin': 'lower'}
    mpl.rcParams.update(settings)


def colorbar(mappable, pos="right", ticks=None):
    ax = mappable.axes
    fig = ax.figure
    if pos == "right":
        orientation = "vertical"
        cax = inset_axes(ax,
                         width="5%",
                         height="100%",
                         loc="lower left",
                         bbox_to_anchor=(1.05, 0, 1, 1),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
    elif pos == "bottom":
        cax = inset_axes(ax,
                         width="100%",
                         height="5%",
                         loc="lower left",
                         bbox_to_anchor=(0, -0.20, 1, 1),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
        orientation = "horizontal"
    elif pos == "top":
        cax = inset_axes(ax,
                         width="100%",
                         height="5%",
                         loc="upper left",
                         bbox_to_anchor=(0, +0.07, 1, 1),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
        orientation = "horizontal"
    cax.ticklabel_format(style='sci', scilimits=(-1,1))
    return fig.colorbar(mappable, cax=cax,    orientation=orientation,
            ticks=ticks)


def PlotWithErrorShades(array,x = None, title=None, color=None, style='-',
        hatch='none'):
    mean = np.mean(np.log(array), axis=0)
    if x is None:
        x = np.arange(len(mean))
    std = np.std(np.log(array), axis=0)
    upper = np.exp(mean+std)
    lower = np.exp(mean-std)
    plt.plot(x, np.exp(mean),style,label=title, color=color, alpha=.8)
    if hatch=='none':
        plt.fill_between(x,upper,lower,alpha=0.3, edgecolor = 'none', facecolor=color) 
    else:
        plt.fill_between(x,upper,lower, edgecolor='w', facecolor='none',
                hatch=hatch, linewidth=0.0) 

def PlotPowSpecs(filename, A, mock_position, KL, 
        ampl = None,
        square = True,
        inftype = ['o','c','pc'],
        ylim = []
        ):
    '''
    Function that plots power spectra with shaded areas or realizations
    Parameters
    ----------
    filename        : string
    A               : Amplitude
    mock_position   : position of ground truth
    KL              : list of KLs
    ampl            : amplitude for reconstructed positions and samples
    square          : boolean, True if input is amplitude and is needed to be
                      squared
    inftype         : array of inference types plotted ('o','c','pc','pca')
    '''
    if ampl is None:
        ampl = A
    col, style, hatches = getPlotSpecifications(inftype)
    title = [
            "Original Posterior Power Spectrum(sampled)",
            "Compressed Posterior Power Spectrum",
            "Patchwise Compressed Posterior Power Spectrum",
            ]
    plt.plot(A(mock_position).to_global_data()**(1+square), color='green', label='Ground Truth Power Spectrum')
    for ii in range(len(KL)):
        for hatch in [['none']*len(hatches),hatches]:
            powers = np.array([ampl(s +
                KL[ii].position).to_global_data()**(1+square) for s in KL[ii].samples])
            PlotWithErrorShades(powers,
                    title=title[ii],
                    color=col[ii],
                    style=style[ii], hatch=hatch[ii])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("harmonic mode", labelpad = 1.)
    plt.ylabel("power spectrum", labelpad = 2.)
    if len(ylim) > 0:
        plt.ylim(ylim)
    plt.savefig(filename)
    print("Saved results as '{}'.".format(filename))
    plt.close()


def PlotNL(fig, x, pltnr=1, label='Ground Truth', cbarticks=None,
        vmin = 0., vmax =1.):
    pltstyle = 'mplstyles/lowerleftplot.mplstyle' if pltnr%2 == 1 else 'mplstyles/lowerrightplot.mplstyle'
    with plt.style.context(pltstyle):
        fig.add_subplot(1,2,pltnr)
        plt.locator_params(nbins=6)
        im = plt.imshow(x,
                extent = [0.,1.,0.,1.],
                vmin = vmin,
                vmax = vmax,
                label=label,
                origin="lower",)
        plt.xlabel("position", labelpad = 1.)
        colorbar(im, "bottom",ticks=cbarticks)

def PlotNLComp(fig, x, pltnr=2, label='Difference to Original Posterior Mean', vmax =1.):
    vmin = - vmax
    pltstyle = 'mplstyles/lowerleftplot.mplstyle' if pltnr%2 == 1 else 'mplstyles/lowerrightplot.mplstyle'
    with plt.style.context(pltstyle):
        mpl.rcParams['image.cmap'] = 'PiYG'
        fig.add_subplot(1,2,pltnr)
        plt.locator_params(nbins=6)
        im = plt.imshow(x,
                extent = [0.,1.,0.,1.],
                vmin = vmin,
                vmax = vmax,
                label=label,
                origin="lower",)
        plt.xlabel("position", labelpad = 1.)
        colorbar(im, "bottom", ticks=[-vmax,-.5*vmax,0.,.5*vmax,vmax])

def PlotSetup(filename, signal, mock_position, R, data):
    with plt.style.context('mplstyles/tuple2dplot.mplstyle'):
        fig = plt.figure()
        PlotNL(fig, signal(mock_position).to_global_data().T)
        PlotNL(fig, R.adjoint_times(data).to_global_data().T,
                label='Back Projection of the Original Data',
                pltnr=2)
        plt.savefig(filename)
        print("Saved results as '{}'.".format(filename))
        plt.close()
 

def PlotMeanStd(filename, mean, var):
    with plt.style.context('mplstyles/tuple2dplot.mplstyle'):
        fig = plt.figure()
        # plot mean
        PlotNL(fig,
                mean.to_global_data().T,
                label='Back Projection of the Original Data',
                pltnr=1)
        # plot std
        PlotNL(fig,
                ift.sqrt(var).to_global_data().T,
                vmin=0.,
                vmax=0.16,
                label='Posterior Standard Deviation',
                pltnr=2,
                cbarticks=[0.,0.04,.08,.12,.16])
        plt.savefig(filename)
        print("Saved results as '{}'.".format(filename))
        plt.close()


def PlotMeanComp(filename, mean, mean_o, vmax2=0.2):
    vmax2 = RoundDigit(vmax2)
    with plt.style.context('mplstyles/tuple2dplot.mplstyle'):
        fig = plt.figure()
        # plot mean
        PlotNL(fig,
                mean.to_global_data().T,
                label='Posterior Mean',
                pltnr=1)
        # plot mean difference
        PlotNLComp(fig, (mean-mean_o).to_global_data().T, pltnr=2,
                label='Difference to Original Posterior Mean',
                vmax = vmax2)
        plt.savefig(filename)
        print("Saved results as '{}'.".format(filename))
        plt.close()

def PlotStdComp(filename, var, var_o, vmax2=7.e-2):
    vmax2 = RoundDigit(vmax2)
    with plt.style.context('mplstyles/tuple2dplot.mplstyle'):
        fig = plt.figure()
        # var
        PlotNL(fig,
                ift.sqrt(var).to_global_data().T,
                label='Posterior Standard Deviation',
                vmin = 0.,
                vmax = .16,
                pltnr=1,
                cbarticks=[0.,.04,.08,.12,.16])
        # var differences
        PlotNLComp(fig,
                (ift.sqrt(var)-ift.sqrt(var_o)).to_global_data().T,
                pltnr=2,
                label='Difference to Original Posterior Standard Deviation',
                vmax = vmax2)
        plt.savefig(filename)
        print("Saved results as '{}'.".format(filename))
        plt.close()


def Einsen(shape, domain, entry, numpy=False):
    einsen = np.zeros(shape)
    einsen[entry] = 1.
    if not numpy:
        einsen = ift.Field(domain, val = einsen)
    return einsen

def PlotEigend(filename, data_shape, data_domain, R, entries=np.arange(4),
        plotsperline = 2):
    nrpl = len(entries)
    nrln = nrpl//plotsperline+nrpl%plotsperline
    aa=['bottom','bottom']
    bb=['mplstyles/lowerleftplot.mplstyle',
            'mplstyles/lowerrightplot.mplstyle',
            'mplstyles/lowerleftplot.mplstyle',
            'mplstyles/lowerrightplot.mplstyle']
    with plt.style.context('mplstyles/tuple2dplot.mplstyle'):
        mpl.rcParams['figure.figsize'] = 3.2*plotsperline, 3.9*nrln
        if nrln >1:
            mpl.rcParams['figure.subplot.bottom']  = 0.11
        else:
            mpl.rcParams['figure.subplot.bottom']  = 0.21
        fig = plt.figure()
        for ii in range(nrpl):
            with plt.style.context(bb[ii%len(bb)]):
                mpl.rcParams['image.cmap'] = 'RdBu'
                fig.add_subplot(nrln,plotsperline,ii+1)
                im = plt.imshow((R.adjoint_times(Einsen(data_shape, data_domain, entries[ii]))).to_global_data().T,
                        extent = [0.,1.,0.,1.],
                        vmin = -6.e-3,
                        vmax = 6.e-3,
                        label='Eigendirection {}'.format(ii),
                        origin="lower",)
                plt.xlabel("position", labelpad = 1.)
                colorbar(im, aa[(ii//2)%len(aa)], ticks=[-6.e-3,-3.e-3,0.,3.e-3,6.e-3])
        plt.savefig(filename)
        print("Saved results as '{}'.".format(filename))
        plt.close()

def PlotEigendNumpy(filename, data_shape, R, entries=np.arange(4)):
    nrpl = len(entries)
    nrln = nrpl//2+nrpl%2
    aa=['bottom','bottom']
    bb=['mplstyles/lowerleftplot.mplstyle',
            'mplstyles/lowerrightplot.mplstyle',
            'mplstyles/lowerleftplot.mplstyle',
            'mplstyles/lowerrightplot.mplstyle']
    with plt.style.context('mplstyles/tuple2dplot.mplstyle'):
        mpl.rcParams['figure.figsize'] = 6.4, 3.7*nrln
        fig = plt.figure()
        for ii in range(nrpl):
            with plt.style.context(bb[ii%len(bb)]):
                mpl.rcParams['image.cmap'] = 'RdBu'
                fig.add_subplot(nrln,2,ii+1)
                im = plt.imshow(R.transpose()(Einsen(data_shape, None,
                    entries[ii], numpy=True)).transpose(),
                        extent = [0.,1.,0.,1.],
                        label='Eigendirection {}'.format(ii),
                        origin="lower",)
                plt.xlabel("position", labelpad = 1.)
                colorbar(im, aa[(ii//2)%len(aa)], ticks=[-6.e-3,-3.e-3,0.,3.e-3,6.e-3])
        plt.savefig(filename)
        print("Saved results as '{}'.".format(filename))
        plt.close()

def PlotBackProj(filename, Rd0, Rd1, ticks2=None):
    with plt.style.context('mplstyles/tuple2dplot.mplstyle'):
        fig = plt.figure()
        # before
        with plt.style.context('mplstyles/lowerleftplot.mplstyle'):
            mpl.rcParams['image.cmap'] = 'RdBu'
            fig.add_subplot(1,2,1)
            im = plt.imshow(Rd0.to_global_data().T,
                        extent = [0.,1.,0.,1.],
                        vmin = -0.16,
                        vmax = 0.16,
                        label='Back Projection of the Compressed Data Before',
                        origin="lower",)
            plt.xlabel("position", labelpad = 1.)
            colorbar(im, "bottom", ticks=[-0.16, -0.08, 0., 0.08,.16])
        # after inference
        with plt.style.context('mplstyles/lowerrightplot.mplstyle'):
            mpl.rcParams['image.cmap'] = 'RdBu'
            fig.add_subplot(1,2,2)
            im2 = plt.imshow(Rd1.to_global_data().T,
                    extent = [0.,1.,0.,1.],
                    vmin = ticks2[0],
                    vmax = ticks2[-1],
                    label='and After the Inference',
                    origin="lower",)
            plt.xlabel("position", labelpad = 1.)
            colorbar(im2, "bottom", ticks=ticks2)
        plt.savefig(filename)
        print("Saved results as '{}'.".format(filename))
        plt.close()

def PlotPoints(filename, points1, points2):
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    for ii in range(len(points1)):
        plt.plot(points1[ii],'o')
    fig.add_subplot(1,2,2)
    for ii in range(len(points2)):
        plt.plot(points2[ii],'o')
    plt.savefig(filename)
    print("Saved results as '{}'.".format(filename))
    plt.close()



with open(nl_Wimmerl+'setup.pk', 'rb') as f:
    filename, signal, A, mock_position, patch_number = pickle.load(f)
with open(nl_Wimmerl+"NonLin_o.pk",'rb') as f:
    KL_o, R, Ninv, data, mean_o, var_o, inftime_o, sumtime_o = pickle.load(f)
with open(nl_Wimmerl+"NonLin_c.pk",'rb') as f:
    KL_c, S_c, R_c, N_c_inv, d_c, mean_c, var_c, R_c0, d_c0, unc, gamma_min_c, inftime_c, sumtime_c = pickle.load(f)
with open(nl_Wimmerl+"NonLin_pc.pk",'rb') as f:
    KL_pc, R_pc, N_pc_inv, d_pc, mean_pc, var_pc, Rd0, Rd1, gamma_min_pc, inftime_pc, sumtime_pc = pickle.load(f)

musqd_pc = N_pc_inv._ldiag
RGCP = RGCheckerboardPatcher(signal.target, patch_number)

def create_vector(domain, intgr):
    arr = np.zeros(domain.shape)
    arr[intgr] = 1.
    return ift.Field(domain, val = arr)

print("Time for usual, compression and patchcomp for n_rep inference steps:")
print(np.mean(inftime_o),np.std(inftime_o))
print(np.mean(inftime_c), np.std(inftime_c))
print(np.mean(inftime_pc), np.std(inftime_pc))

print("Totel time for usual, compression and patchcomp method and inference:")
print(sumtime_o)
print(sumtime_c)
print(sumtime_pc)

print("\gamma_min for compressed and patchwise compressed method:")
print(gamma_min_c)
print([np.mean(gamma_min_c[ii]) for ii in range(len(gamma_min_c))])
gamma_min_pc = np.concatenate(gamma_min_pc)
gamma_min_pc = gamma_min_pc.reshape((gamma_min_pc.size//2,2))
isvalidpatch = [1,3,4,6,9,11,12,14]
print(gamma_min_pc)
print(np.mean(gamma_min_pc[isvalidpatch,0]), np.mean(gamma_min_pc[isvalidpatch,1]))
print(np.std(gamma_min_pc[isvalidpatch,0]),np.std(gamma_min_pc[isvalidpatch,1]))

print("data size (o,c,pc)")
print(data.domain.shape)
print(d_c.domain.shape)
print(d_pc.domain.shape)

filename = filename.replace("../Plots","Plots")

#Plot setup (synthsignal and data)
PlotSetup(filename.format("Setup"), signal, mock_position, R, data)
#Plot posterior mean and stddev for
#original data
PlotMeanStd(filename.format("o"), mean_o, var_o)
#compressed data
PlotMeanComp(filename.format("mean_c"), mean_c, mean_o)
PlotStdComp(filename.format("std_c"), var_c, var_o, vmax2=7.e-2)
#patchwise compressed data
PlotMeanComp(filename.format("mean_pc"), mean_pc, mean_o)
PlotStdComp(filename.format("std_pc"), var_pc, var_o, vmax2 = 7.e-2)


def correlated_field_nl():
    ps_dim = np.array([128,128])
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
    return ht(vol*power_distributor(A)*xi)

def power_analyze(field):
    fft = ift.HartleyOperator(field.domain[0].get_default_codomain(),
            target=field.domain[0])
    xx = fft.inverse_times(field)
    return ift.sugar.power_analyze(xx)

def ampl_nl():
    def amp(x):
        return power_analyze(correlated_field_nl()(x))
    return amp

PlotPowSpecs(filename.format("PowSpecs"), ampl_nl(), mock_position, [KL_o, KL_c, KL_pc],
        square = False,
        ampl = ampl_nl(),
        inftype = ['o','c','pc'])

#Plot eigendirections of
#compressed recon before and after
PlotEigend(filename.format("R_c"), d_c.shape, d_c.domain,R_c)
PlotEigend(filename.format("R_c1"), d_c.shape, d_c.domain,R_c, entries=[0,1])
PlotEigend(filename.format("R_c2"), d_c.shape, d_c.domain,R_c, entries=[2,3])
#patchwise compressed reconstruction before and after
entries = [(1,0),(3,0),(4,0),(6,0),(9,0),(11,0),(12,0),(14,0)]
PlotEigend(filename.format("R_pc"), d_pc.shape, d_pc.domain, R_pc @ RGCP, entries)

#Plot backprojection
#compressed data
PlotBackProj(filename.format("BP_c"), R_c0.adjoint_times(d_c0),
        R_c.adjoint_times(d_c),
        ticks2=[-30.,-15.,0.,15.,30.])
#patchwise compressed data
PlotBackProj(filename.format("BP_pc"), Rd0, Rd1, ticks2=[-10.,-5.,0.,5.,10.])

# Plot mean:
for ll in range(n_comp):
    if ll < n_comp-1:
        with open("Wimmerl/NonLin_c_{}.pk".format(ll),'rb') as f:
            S_cl, R_cl, N_c_invl, d_cl, mean_cl =  pickle.load(f)
    else:
        S_cl, R_cl, N_c_invl, d_cl, mean_cl = S_c, R_c, N_c_inv, d_c, mean_c
    N_cl = N_c_invl.inverse
    matr = R_cl @ S_cl @ R_cl.adjoint + N_cl
    SD_rv_inv = ift.ScalingOperator(1., S_cl.domain) + S_cl @ R_cl.adjoint @ N_c_invl @ R_cl
    IC=ift.GradientNormController(iteration_limit = 500, tol_abs_gradnorm =1e-3)
    SD_rv_inv = ift.InversionEnabler(SD_rv_inv, IC)
    matr = ift.InversionEnabler(matr, IC, approximation=N_cl)
    m_r = S_cl(R_cl.adjoint_times(matr.inverse(d_cl-R_cl(mean_cl))))
    m_r = m_r + mean_cl
    PlotMeanComp(filename.format("mean_rv_{}".format(ll)), m_r,
            mean_cl,vmax2=np.max(np.abs((m_r-mean_cl).to_global_data())))
# Plot variance:
sc_r = ift.StatCalculator()
N_c = N_c_inv.inverse
matr = R_c @ S_c @ R_c.adjoint + N_c
matr = ift.InversionEnabler(matr, IC, approximation=N_c)
# start loop
for ii in range(N_samples):
    s_r = S_c.draw_sample()
    d_r = R_c(s_r) + N_c.draw_sample()
    m_r = S_c(R_c.adjoint_times(matr.inverse(d_r)))
    sc_r.add( s_r - m_r)
unc = sc_r.var
PlotStdComp(filename.format("ResponseVariance"), unc, var_c,
        vmax2=np.max([1.e-2,np.max(np.abs((unc-var_c).to_global_data()))]))

#####################
# Radio Application #
#####################

def Rescale(mean):
    xshift = .15
    yshift = 1.
    distances = mean.domain[0].distances[0] #[rad]
    distances *= 180./np.pi*60. #transform rad to Grad to arcmin
    extent0 = mean.shape[0]*distances/2.
    extent1 = extent0
    extent = [-extent0+xshift,extent0+xshift,-extent1+yshift,extent1+yshift]
    #rescale from Jy/sr to Jy/arcmin^2
    rescaled_mean = ((np.pi/(180*60))**2)*mean.to_global_data().T 
    return rescaled_mean, extent 

def PlotRslvMean(fig, mean, pltnr=1, vmax=None, label='Posterior Mean',
        cbarticks=None):
    rescaled_mean, extent = Rescale(mean)
    mpl.rcParams['image.cmap'] = 'afmhot'
    pltstyle = 'mplstyles/lowerleftplot.mplstyle' if pltnr%2 == 1 else 'mplstyles/lowerrightplot.mplstyle'
    with plt.style.context(pltstyle):
        fig.add_subplot(1,2,pltnr)
        plt.locator_params(nbins=6)
        im = plt.imshow(rescaled_mean,
                extent = extent,
                vmin = 0.,
                vmax = vmax,
                label=label,
                origin="lower",)
        plt.xlabel("position [arcmin]", labelpad = 1.)
        plt.xlim(-3.,3.)
        plt.ylim(-3.,3.)
        colorbar(im, "bottom",ticks=cbarticks)

def PlotRslvComp(fig, vardif, pltnr=2, vmin=None, vmax=None,label='Difference to Original Posterior Standard Deviation'):
    rescaled_vardif, extent = Rescale(vardif)
    with plt.style.context('mplstyles/lowerrightplot.mplstyle'):
        mpl.rcParams['image.cmap'] = 'PiYG'
        ax = fig.add_subplot(1,2,pltnr)
        plt.locator_params(nbins=6)
        im2 = plt.imshow(rescaled_vardif,
                extent=extent,
                vmin = vmin,
                vmax = vmax,
                label=label,
                origin="lower",)
        plt.xlabel("position [arcmin]", labelpad = 1.)
        plt.xlim(-3.,3.)
        plt.ylim(-3.,3.)
        ticks = None if vmax is None else [-vmax,-.5*vmax,0.,.5*vmax,vmax]
        colorbar(im2, "bottom", ticks= ticks)

def PlotRslvMeanStd(filename, mean, var, vmax2=None):
    with plt.style.context('mplstyles/tuple2dplot.mplstyle'):
        fig = plt.figure()
        # plot mean
        PlotRslvMean(fig, mean,pltnr=1)
        # plot std
        PlotRslvMean(fig, ift.sqrt(var),pltnr=2,
                vmax=vmax2,
                label='Posterior Standard Deviation')
        plt.savefig(filename)
        print("Saved results as '{}'.".format(filename))
        plt.close()

def PlotRslvMeanComp(filename, mean, mean_o, vmax2=None):
    vmin2 = -vmax2 if vmax2 is not None else None
    distances = mean.domain[0].distances[0]
    extent0 = mean_pc.shape[0]*distances*1e3
    extent1 = extent0
    with plt.style.context('mplstyles/tuple2dplot.mplstyle'):
        fig = plt.figure()
        # plot mean
        PlotRslvMean(fig, mean,pltnr=1)
        PlotRslvComp(fig, mean-mean_o, vmin=vmin2,
                vmax=vmax2,
                label='Difference to Original Posterior Mean',
                pltnr=2)
        plt.savefig(filename)
        print("Saved results as '{}'.".format(filename))
        plt.close()

def PlotRslvStdComp(filename, var, var_o, vmax1=None, vmax2=None):
    vmax1 = RoundDigit(vmax1) if vmax1 is not None else None
    vmin2 = -vmax2 if vmax2 is not None else None
    distances = var.domain[0].distances[0]
    extent0 = var.shape[0]*distances*1e3
    extent1 = extent0
    with plt.style.context('mplstyles/tuple2dplot.mplstyle'):
        fig = plt.figure()
        # plot std
        PlotRslvMean(fig, ift.sqrt(var),pltnr=1,
                vmax=vmax1,
                label='Posterior Standard Deviation')
        PlotRslvComp(fig, ift.sqrt(var)-ift.sqrt(var_o), vmin=vmin2,
                vmax=vmax2,
                label='Difference to Original Posterior Standard Deviation')
        plt.savefig(filename)
        print("Saved results as '{}'.".format(filename))
        plt.close()

def PlotRslvPowSpecs(filename, pspec, position, samples,
        ylim=[],
        ampl = None,
        inftype = ['o','pc'],
        ):
    colors, style, hatches = getPlotSpecifications(inftype)
    if ampl is None:
        ampl = pspec
    x = pspec.target[0].k_lengths*ARCMIN2RAD
    for hatch in [['none']*len(hatches),hatches]:
        for ii in range(len(position)):
            powers = np.array([ampl(sample +
                position[ii]).to_global_data() for sample in
                samples[ii]])
            PlotWithErrorShades(powers,
                    x=x,
                    title="Compressed Posterior Power Spectrum",
                    color=colors[ii],
                    style=style[ii],
                    hatch=hatch[ii])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("harmonic mode [arcmin${}^{-1}$]", labelpad = 1.)
    plt.ylabel("power spectrum [1]", labelpad = 2.)
    if len(ylim) > 0:
        plt.ylim(ylim)
    plt.savefig(filename)
    print("Saved results as '{}'.".format(filename))
    plt.close()

def PlotToC(filename, mean_o, mean_c, uv, d_c, markersize=1):
    with plt.style.context('mplstyles/toc.mplstyle'):
        mpl.rcParams['figure.dpi'] = '300'
        fig = plt.figure()
        mpl.rcParams['figure.figsize'] = str(55*0.03937008)+', '+str(50*0.03937008)
        rescaled_mean, extent = Rescale(mean_o)
        mpl.rcParams['image.cmap'] = 'afmhot'
        fig.add_subplot(2,2,2)
        plt.locator_params(nbins=6)
        im = plt.imshow(rescaled_mean,
                extent = extent,
                vmin = 0.,
                #vmax = vmax,
                origin="lower",)
        plt.xlim(-3.,3.)
        plt.ylim(-3.,3.)
        plt.title('reconstruction')
        plt.xlabel("position [arcmin]", labelpad = 1.)
        rescaled_mean, extent = Rescale(mean_c)
        mpl.rcParams['image.cmap'] = 'afmhot'
        fig.add_subplot(2,2,4)
        plt.locator_params(nbins=6)
        im = plt.imshow(rescaled_mean,
                extent = extent,
                vmin = 0.,
                #vmax = vmax,
                origin="lower",)
        plt.xlim(-3.,3.)
        plt.ylim(-3.,3.)
        plt.title('reconstruction')
        plt.xlabel("position [arcmin]", labelpad = 1.)
        leftPlots = {
                'ytick.right'         :   False,
                'ytick.left'         :   True,
                'ytick.labelright'   :   False,
                'ytick.labelleft'    :   True,
                }
        mpl.rcParams.update(leftPlots)
        fig.add_subplot(2,2,1)
        plt.scatter(uv[:, 0], uv[:, 1],
                marker='.',
                s=markersize,
                edgecolors='none')
        plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2),useMathText=True)
        plt.xlabel('u [arcmin${}^{-1}$]')
        plt.ylabel('v [arcmin${}^{-1}$]')
        plt.title('original data')
        fig.add_subplot(2,2,3)
        plt.scatter(np.arange(len(d_c)), d_c,
                marker='.',
                s=markersize,
                edgecolors='none')
        plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2),useMathText=True)
        mpl.rcParams.update(leftPlots)
        plt.xlabel('index')
        plt.ylabel('value')
        plt.title('compressed data')
    plt.savefig(filename)
    print("Saved results as '{}'.".format(filename))
    plt.close()


def PlotChessPatches(uv, patchnumber=[64, 64], markersize=5,
        filename = 'Plots/PatchedData.png',
        linewidth=0.3):
    mpl.rcParams['figure.figsize'] = str(figwidth)+', '+str(figheight)
    mpl.rcParams['figure.dpi'] = '300'
    minu = np.amin(uv[:, 0])
    maxu = np.amax(uv[:, 0])
    minv = np.amin(uv[:, 1])
    maxv = np.amax(uv[:, 1])
    plt.scatter(uv[:, 0], uv[:, 1],
                marker='.',
                s=markersize,
                edgecolors='none')
    for asdf in range(patchnumber[0] + 1):
        for fdsa in range(patchnumber[1] + 1):
            plt.plot([minu + np.float64(asdf)*(maxu - minu)/patchnumber[0], minu + np.float64(asdf)*(maxu - minu)/patchnumber[0]],
                     [minv, maxv], 'k', linewidth=linewidth, alpha=0.1)
            plt.plot([minu, maxu], [minv + np.float64(fdsa)*(maxv - minv)/patchnumber[1], minv + np.float64(fdsa)*(maxv - minv)/patchnumber[1]],
                     'k', linewidth=linewidth,alpha=0.1)
    plt.xlabel('u [arcmin${}^{-1}$]')
    plt.ylabel('v [arcmin${}^{-1}$]')
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2),useMathText=True)
    plt.xlim(minu, maxu)
    plt.ylim(minv, maxv)
    plt.savefig(filename)
    print("Plotted PatchedData in subfolder /Plots/")
    plt.close()

with open(rs_Wimmerl+"{}-sc_bs.pk".format(current_time), 'rb') as f:
    filename_pc, position_pc, samples_pc, mean_pc, var_pc = pickle.load(f)
with open("Wimmerl/sc_o_9.pk", 'rb') as f:
    filename_o, position_o, samples_o, pspec, mean_o, var_o, rcounter_o, inftime_o, sumtime_o = pickle.load(f)

PlotRslvMeanComp("./Plots/{}_rslv_mean_comp.pdf".format(current_time), mean_pc,
        mean_o, vmax2 = 6)
PlotRslvStdComp("./Plots/{}_rslv_std_comp.pdf".format(current_time), var_pc,
        var_o, vmax2 = 3)

PlotRslvMeanStd(filename_o.format("Rec"), mean_o, var_o)
PlotRslvMeanStd(filename_o.format("Rec_scaled"), mean_o, var_o, vmax2=5.)
PlotRslvMeanStd(filename_o.format("Rec_pc"), mean_pc, var_pc, vmax2=5.)

samples_o_arr = [sample.extract(pspec.domain) for sample in samples_o]
samples_pc_arr = [sample.extract(pspec.domain) for sample in samples_pc]
PlotRslvPowSpecs(filename_o.format("PowSpec"), pspec,
        [position_o.extract(pspec.domain), position_pc.extract(pspec.domain)],
        [samples_o_arr, samples_pc_arr],
        inftype=['o','pc'])

with open(rs_Wimmerl+"{}-times2.pk".format(current_time), 'rb') as f:
    Rcounter, inftime_pc,sumtime_pc, gamma_min = pickle.load(f)

print("\gamma_min for rslv:")
gamma_min = np.concatenate(gamma_min)
gamma_min = gamma_min.reshape((gamma_min.size//2,2))
print(gamma_min.shape)
print(np.mean(gamma_min[:,0]), np.mean(gamma_min[:,1]))
print(np.std(gamma_min[:,0]), np.std(gamma_min[:,1]))
print(np.max(gamma_min[:,0]), np.min(gamma_min[:,1]))

with open(rs_Wimmerl+"{}-BigSparseResponse.pk".format(current_time), 'rb') as f:
    R_c = pickle.load(f)

def GetUVsky():
    np.random.seed(42)
    dht = rve.DataHandler((256, 256),
                          rve.DEG2RAD*np.array((0.24, 0.24)),
                          200000,
                          shift=(0, 0.1*rve.DEG2RAD),
                          datamode="vis")
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
    uv[:,0] = uv[:,0]*dht._freqs[0]/SPEEDOFLIGHT
    uv[:,1] = uv[:,1]*dht._freqs[0]/SPEEDOFLIGHT
    #transform 1/rad to Grad to 1/arcmin
    uv[:,0] = uv[:,0]/(180./np.pi*60.)
    uv[:,1] = uv[:,1]/(180./np.pi*60.)
    imgr.init_diffuse(**dct)
    sky = imgr.sky
    return d, uv, sky
def Getdc(current_time):
    with open(rs_Wimmerl+"{}-npdc_current.pk".format(current_time), 'rb') as f:
        npdc_current = pickle.load(f)
    npdc = np.concatenate(npdc_current)
    return npdc

_, uv, sky = GetUVsky()

def pow_an_log_sky():
    def f(field):
        return power_analyze(ift.log(sky(field)))
    return f

PlotRslvPowSpecs(filename_o.format("PowAnaSpec"), pspec,
        [position_o, position_pc],
        [samples_o, samples_pc],
        ampl = pow_an_log_sky(),
        inftype=['o','pc'])

d_c = Getdc(current_time)
PlotToC("Plots/toc-image.png", mean_o, mean_pc, uv, d_c)
PlotChessPatches(uv, patchnumber=[64, 64])

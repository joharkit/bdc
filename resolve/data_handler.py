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
# Copyright(C) 2019 Max-Planck-Society
# Author: Philipp Arras

import time

import astropy.io.fits as pyfits
import h5py
import nifty5 as ift
import numpy as np
from astropy.time import Time

from .constants import SPEEDOFLIGHT
from .gridder import GridderMaker


def load_from_hdf5(fname):
    d = {}
    with h5py.File(fname, 'r') as f:
        for kk, vv in f.items():
            d[kk] = vv[:]
    freqs = d['freqs']
    pol = d['pol']
    directions = d['directions']
    sourcenames = d['sourcenames']
    trange = d['trange']
    telescope = d['telescope'][0]
    observer = d['observer'][0]
    del (d['freqs'])
    del (d['pol'])
    del (d['directions'])
    del (d['sourcenames'])
    del (d['trange'])
    del (d['telescope'])
    del (d['observer'])
    return d, freqs, pol, directions, sourcenames, trange, telescope, observer


def _extend(a, n, axis=2):
    if axis == 2:
        return np.repeat(a[:, :, None], n, axis=2)
    if axis == 1:
        return np.repeat(a[:, None], n, axis=1)
    raise NotImplementedError


def _apply_mask(mask, dct):
    return {key: arr[mask] for key, arr in dct.items()}


class DataHandler:
    def __init__(self, shape, fov, rows=-1, eps=1e-7, shift=None, datamode="vis", selection=None):
        ndata = int(rows)
        eps = float(eps)

        sky_space = ift.RGSpace(shape, distances=np.array(fov)/np.array(shape))
        var = np.load('var.npy')
        vis = np.load('vis.npy')
        uvw = np.load('uvw.npy')
        flag = np.load('flags.npy')
        freqs = np.load('freq.npy')

        # Apply flags
        # Data are flagged bad if the FLAG array element is True.
        # https://casa.nrao.edu/Memos/229.html
        # A data point is only taken if all correlations are not flagged.
        # FIXME Proper tracking of flags for each polarization
        flag = np.any(flag, axis=2)

        # Calculate channel factors
        self._nch = len(freqs)
        self._freqs = freqs

        # Take random subset of data points if specified
        len_vis = vis.shape[0]
        if ndata == -1 or (ndata > 0 and ndata > len_vis):
            self._sel = None
            print('Take all data points.')
        else:
            if selection is not None:
                sel = selection
            else:
                sel = np.random.choice(len_vis, ndata)
            self._sel = sel
            uvw = uvw[sel]
            vis = vis[sel]
            var = var[sel]
            flag = flag[sel]

        print('Use (u,v,w)->(-u,-v,-w) symmetry')
        sel = uvw[:, 1] < 0
        uvw[sel] *= -1
        vis[sel] = vis[sel].conjugate()

        if shift is not None and np.sum(np.abs(shift)) > 0:
            shift = -2j*np.pi*np.array(shift)
            shift = np.outer(shift, self._freqs/SPEEDOFLIGHT)
            c = np.exp(np.outer(uvw.T[0], shift[0]) + np.outer(uvw.T[1], shift[1]))
            vis = (c.T*vis.T).T
        vis = np.transpose(vis, (2, 0, 1))
        var = np.transpose(var, (2, 0, 1))
        assert vis.shape == var.shape

        self._R_rest = []
        self._R_degridder = []
        self._R = []
        self._gms = []
        self._vis = []
        self._invvar = []

        vis = np.sum(vis, axis=0)
        var = np.sum(var, axis=0)

        self._eps = eps
        self._datamode = datamode
        gm = GridderMaker(sky_space, uvw, self._freqs, flag, eps=eps, datamode=datamode)
        rest = gm.getRest().adjoint.scale(sky_space.scalar_dvol)

        idx = gm._idx
        self._rawvis = vis[idx]
        self._rawvar = var[idx]
        self._rawuvw = uvw[idx]
        self._rawflag = flag[idx]

        self._R_rest = rest
        self._gm = gm
        self._idx = gm._idx
        self._R_degridder = gm.getGridder().adjoint

        if datamode == "vis":
            tgt = self.R().target
            tmp_vis = np.empty(tgt.shape, dtype=vis.dtype)
            tmp_invvar = np.empty(tgt.shape, dtype=var.dtype)
            tmp_vis = gm.ms2vis(vis)
            tmp_invvar = 1/gm.ms2vis(var).real
            self._vis = ift.from_global_data(tgt, tmp_vis)
            self._invvar = ift.from_global_data(tgt, tmp_invvar)
        elif datamode == "ms":
            tgt = self.R.target
            self._vis = ift.from_global_data(tgt, vis)
            self._invvar = ift.from_global_data(tgt, 1./var)
        self._uvw = uvw

    def j(self):
        rs, invvars, viss = self.get_Rinvvarvis_iter()
        j = next(rs).adjoint(next(viss)*next(invvars))
        for r, vis, invvar in zip(rs, viss, invvars):
            j = j + r.adjoint(vis*invvar)
        return j

    ###########################################################################
    # Getters and setters
    ###########################################################################
    @property
    def sky_domain(self):
        return self.R().domain

    def gm(self):
        return self._gm

    def R(self):
        return self._R_degridder @ self._R_rest

    def R_degridder(self):
        return self._R_degridder

    def R_rest(self):
        return self._R_rest

    def invvar(self):
        return self._invvar

    def vis(self):
        return self._vis

    def select_from_datamask(self, data_mask):
        uvw = self._rawuvw[data_mask]
        flag = self._rawflag[data_mask]
        assert len(self._freqs) == 1
        dom = self.R().domain
        gm = GridderMaker(dom, uvw, self._freqs, flag, eps=self._eps, datamode=self._datamode)
        R_degridder = gm.getGridder().adjoint
        tgt = R_degridder.target
        vis = self._rawvis[data_mask]
        var = self._rawvar[data_mask]
        tmp_vis = np.empty(tgt.shape, dtype=vis.dtype)
        tmp_invvar = np.empty(tgt.shape, dtype=var.dtype)
        tmp_vis = gm.ms2vis(vis)
        tmp_invvar = 1/gm.ms2vis(var).real
        vis = ift.from_global_data(tgt, tmp_vis)
        invvar = ift.from_global_data(tgt, tmp_invvar)
        return R_degridder, ift.makeOp(invvar), vis

    @property
    def freqs(self):
        return self._freqs

    def writefits(self, field, file_name):
        dom = field.domain[0]

        h = pyfits.Header()
        h['BUNIT'] = 'Jy/rad'

        # FIXME Take shift into account
        h['CTYPE1'] = 'RA---SIN'
        h['CRVAL1'] = self._phase_center[0]*180/np.pi
        h['CDELT1'] = -dom.distances[0]*180/np.pi
        h['CRPIX1'] = dom.shape[0]/2
        h['CUNIT1'] = 'deg'
        h['CTYPE2'] = 'DEC---SIN'
        h['CRVAL2'] = self._phase_center[1]*180/np.pi
        h['CDELT2'] = dom.distances[1]*180/np.pi
        h['CRPIX2'] = dom.shape[1]/2
        h['CUNIT2'] = 'deg'

        h['OBJECT'] = self._sourcename
        h['DATE-OBS'] = Time(self._trange[0]/86400.0,
                             scale="utc",
                             format='mjd').iso.split()[0]
        h['DATE-MAP'] = Time(time.time(), format='unix').iso.split()[0]
        h['OBSERVER'] = self._observer
        h['TELESCOP'] = self._telescope

        # FIXME Where does this value come from?
        h['EQUINOX'] = 1979.9
        h['EQUINOX'] = 2000

        # FIXME Add Prior parameters and minimization history
        hdu = pyfits.PrimaryHDU(field.to_global_data().T, header=h)
        hdulist = pyfits.HDUList([hdu])
        hdulist.writeto(file_name, overwrite=True)

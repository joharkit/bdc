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
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.
# Author: Philipp Arras, Martin Reinecke

import nifty5 as ift
import nifty_gridder


class GridderMaker(object):
    def __init__(self,
                 dirty_domain,
                 uvw,
                 freq,
                 flags,
                 eps=2e-13,
                 wrange=None,
                 datamode="ms",
                 idx=None):
        self._datamode=datamode
        dirty_domain = ift.makeDomain(dirty_domain)
        if (len(dirty_domain) != 1
                or not isinstance(dirty_domain[0], ift.RGSpace)
                or not len(dirty_domain.shape) == 2):
            raise ValueError("need dirty_domain with exactly one 2D RGSpace")
        pixsize_x, pixsize_y = dirty_domain[0].distances
        if freq.ndim != 1:
            raise ValueError("freq must be a 1D array")
        bl = nifty_gridder.Baselines(coord=uvw, freq=freq)
        nxdirty, nydirty = dirty_domain.shape
        self._gconf = nifty_gridder.GridderConfig(nxdirty=nxdirty,
                                                  nydirty=nydirty,
                                                  epsilon=eps,
                                                  pixsize_x=pixsize_x,
                                                  pixsize_y=pixsize_y)
        nu = self._gconf.Nu()
        nv = self._gconf.Nv()
        dct = {
            'baselines': bl,
            'gconf': self._gconf,
            'flags': flags
        }
        if wrange is not None:
            assert len(wrange) == 2
            if idx is not None:
                self._idx = idx
            else:
                self._idx = nifty_gridder.getIndices(**dct, wmin=wrange[0], wmax=wrange[1])
            self._w = wrange[0]
        else:
            if idx is not None:
                self._idx = idx
            else:
                self._idx = nifty_gridder.getIndices(**dct)
            self._w = None
        self._bl = bl
        self.n_sampling_points = len(self._idx)
        self._grid_domain = ift.RGSpace([nu, nv],
                                        distances=[1, 1],
                                        harmonic=False)
        self._dirty_domain = dirty_domain

    def getGridder(self):
        return RadioGridder(self._grid_domain, self._bl, self._gconf,
                            self._idx, self._w is not None, datamode=self._datamode)

    def getRest(self):
        return _RestOperator(self._dirty_domain, self._grid_domain,
                             self._gconf, w=self._w)

    def getFull(self):
        return self.getRest() @ self._gridder

    def ms2vis(self, x):
        return self._bl.ms2vis(x, self._idx)


class _RestOperator(ift.LinearOperator):
    def __init__(self, dirty_domain, grid_domain, gconf, w=None):
        """w None: No w stacking, w 0: still apply 1/n term"""
        self._domain = ift.makeDomain(grid_domain)
        self._target = ift.makeDomain(dirty_domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

        if w is None:
            self._f = gconf.grid2dirty
            self._fadj = gconf.dirty2grid
        else:
            self._f = lambda x: gconf.apply_wscreen(gconf.grid2dirty_c(x), w,
                                                    True).real
            self._fadj = lambda x: gconf.dirty2grid_c(
                gconf.apply_wscreen(x, w, False))

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = x.to_global_data()
        f = self._f if mode == self.TIMES else self._fadj
        return ift.from_global_data(self._tgt(mode), f(res))


class RadioGridder(ift.LinearOperator):
    def __init__(self, grid_domain, bl, gconf, idx, complex_mode=False, datamode="ms"):
        if datamode == "vis":
            self._domain = ift.DomainTuple.make(ift.UnstructuredDomain(len(idx)))
        else:
            self._domain = ift.DomainTuple.make(ift.UnstructuredDomain((len(idx),1)))
        self._target = ift.DomainTuple.make(grid_domain)
        self._bl = bl
        self._gconf = gconf
        self._idx = idx
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._complex_mode = bool(complex_mode)
        self._datamode = datamode

    def apply(self, x, mode):
        self._check_input(x, mode)
        # FIXME Whiten visibilities!
        if mode == self.TIMES:
            if self._datamode == "vis":
                f = nifty_gridder.vis2grid
                if self._complex_mode:
                    f = nifty_gridder.vis2grid_c
            else:
                f = nifty_gridder.ms2grid
                if self._complex_mode:
                    f = nifty_gridder.ms2grid_c
        else:
            if self._datamode == "vis":
                f = nifty_gridder.grid2vis #Ersetze durch grid2ms Funktionen
                if self._complex_mode:
                    f = nifty_gridder.grid2vis_c
            else:
                f = nifty_gridder.grid2ms #Ersetze durch grid2ms Funktionen
                if self._complex_mode:
                    f = nifty_gridder.grid2ms_c
        res = f(self._bl, self._gconf, self._idx, x.to_global_data())
        return ift.from_global_data(self._tgt(mode), res)

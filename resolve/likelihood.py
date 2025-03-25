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

from functools import reduce
from operator import add, mul

import nifty5 as ift

from .sugar import tuple_to_image, zero_to_nan


class LikelihoodMaker:
    def __init__(self, dh, sky, calibration_operators={}, alpha=None, q=None):
        self._sky = sky
        self._cal_ops = calibration_operators
        self._dh = dh

        if isinstance(self._sky, ift.Field):
            self._const_sky = True
        elif isinstance(self._sky, ift.Operator):
            self._const_sky = False
        else:
            raise TypeError
        if len(self._cal_ops) > 0:
            self._calibration = True
        else:
            self._calibration = False

    @property
    def calibration_operators(self):
        return self._cal_ops

    def information_source(self, position=None):
        if position is not None:
            if self._dh.active_wplanes != 1:
                raise NotImplementedError
            vis, invvar = self.calibrated_visinvvar(position)
            return self._dh.R(0).adjoint(vis*invvar)
        return self._dh.j()

    def signal_response(self, wplane=0):
        rsky = self._dh.R(wplane)(self._sky)
        if self._calibration:
            cop = self.make_calibration()
            if self._const_sky:
                return ift.makeOp(rsky) @ cop
            else:
                return cop*rsky
        else:
            if self._const_sky:
                raise NotImplementedError
            else:
                return rsky

    def get_full(self):
        dh = self._dh
        e = ift.GaussianEnergy(inverse_covariance=ift.makeOp(dh.invvar()), mean=dh.vis()) @ dh.R()
        return e @ self._sky

    def residual_image(self, position):
        dh = self._dh
        res = ift.full(dh.sky_domain, 0)
        for ww in range(dh.active_wplanes):
            sigresp = self.signal_response(ww).force(position)
            res = res + dh.R(ww).adjoint((sigresp - dh.vis(ww))*dh.invvar(ww))
        return res

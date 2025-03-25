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

import numpy as np

import nifty5 as ift


def LAmplitude(target, n_pix, sm, sv, im, iv, key):
    et = ift.ExpTransform(target, n_pix)
    dom = et.domain[0]
    sl = ift.SlopeOperator(dom)
    mean = np.array([sm, im + sm*dom.t_0[0]])
    sig = np.array([sv, iv])
    mean = ift.Field.from_global_data(sl.domain, mean)
    sig = ift.Field.from_global_data(sl.domain, sig)
    linear = sl @ ift.Adder(mean) @ ift.makeOp(sig).ducktape(key)
    op = et @ (0.5*linear).exp()
    return _zmmaskit(op)


def SLAmplitude(target, n_pix, a, k0, sm, sv, im, iv, keys):
    op = ift.LinearSLAmplitude(
        target=target,
        n_pix=n_pix,
        a=a,
        k0=k0,
        sm=sm,
        sv=sv,
        im=im,
        iv=iv,
        keys=keys).clip(-100, 100).exp()
    return _zmmaskit(op)


def _zmmaskit(amplitude):
    mask = np.ones(amplitude.target.shape)
    mask[0] = 0
    mask = ift.makeOp(ift.from_global_data(amplitude.target, mask))
    fld = np.zeros(amplitude.target.shape)
    fld[0] = 1
    adder = ift.Adder(ift.from_global_data(amplitude.target, fld))
    return adder @ mask(amplitude)

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
import scipy as sp

import nifty5 as ift

from .constants import ARCMIN2RAD, SPEEDOFLIGHT


def _vla_eval_poly(coeffs, xs):
    assert coeffs.shape == (3,)
    ys = np.ones_like(xs)
    ys += 1e-3*coeffs[0]*xs**2
    ys += 1e-7*coeffs[1]*xs**4
    return ys + 1e-10*coeffs[2]*xs**6


def gaussian_primary_beam(dom, freq, diameter):
    xf, xp = dom.distances[0]*dom.shape[0]/2, dom.shape[0]
    yf, yp = dom.distances[1]*dom.shape[1]/2, dom.shape[1]
    xx, yy = np.meshgrid(
        np.linspace(-xf, xf, xp), np.linspace(-yf, yf, yp), indexing='ij')
    r = np.sqrt(xx**2 + yy**2)
    lam = SPEEDOFLIGHT/freq
    fwhm = 1.22*lam/diameter
    sigma = fwhm/2.355
    beam = np.exp(-r**2/2/sigma**2)
    return ift.from_global_data(dom, beam)


def bessel_primary_beam(dom, freq, diameter, diameter_blockage=0):
    xf, xp = dom.distances[0]*dom.shape[0]/2, dom.shape[0]
    yf, yp = dom.distances[1]*dom.shape[1]/2, dom.shape[1]
    xx, yy = np.meshgrid(
        np.linspace(-xf, xf, xp), np.linspace(-yf, yf, yp), indexing='ij')
    sinr = np.sin(np.sqrt(xx**2 + yy**2))
    lam = SPEEDOFLIGHT/freq
    x = np.pi/lam*diameter*sinr
    beam = (sp.special.j1(x)/x)*diameter**2
    if diameter_blockage > 0:
        x = np.pi/lam*diameter_blockage*sinr
        beam -= (sp.special.j1(x)/x)*diameter_blockage**2
    beam *= beam
    beam /= np.max(beam)
    return ift.from_global_data(dom, beam)


def vla_primary_beam(dom, freq):
    # freq in MHz
    # Values taken from EVLA memo 195
    dom = ift.DomainTuple.make(dom)
    assert len(dom) == 1
    dom = dom[0]
    # FIXME This table is not complete!
    coeffs = np.array([
        [1040, -1.529, 8.69, -1.88],
        [1104, -1.486, 8.15, -1.68],
        [1168, -1.439, 7.53, -1.45],
        [1232, -1.450, 7.87, -1.63],
        [1296, -1.428, 7.62, -1.54],
        [1360, -1.449, 8.02, -1.74],
        [1424, -1.462, 8.23, -1.83],
        [1488, -1.455, 7.92, -1.63],
        [1552, -1.435, 7.54, -1.49],
        [1680, -1.443, 7.74, -1.57],
        [1744, -1.462, 8.02, -1.69],
        [1808, -1.488, 8.38, -1.83],
        [1872, -1.486, 8.26, -1.75],
        [1936, -1.469, 7.93, -1.62],
        [2000, -1.508, 8.31, -1.68],
        [2052, -1.429, 7.52, -1.47],
        [2180, -1.389, 7.06, -1.33],
        [2436, -1.377, 6.90, -1.27],
        [2564, -1.381, 6.92, -1.26],
        [2692, -1.402, 7.23, -1.40],
        [2820, -1.433, 7.62, -1.54],
        [2948, -1.433, 7.46, -1.42],
        [3052, -1.467, 8.05, -1.70],
        [3180, -1.497, 8.38, -1.80],
        [3308, -1.504, 8.37, -1.77],
        [3436, -1.521, 8.63, -1.88],
        [3564, -1.505, 8.37, -1.75],
        [3692, -1.521, 8.51, -1.79],
        [3820, -1.534, 8.57, -1.77],
        [3948, -1.516, 8.30, -1.66],
        [4052, -1.406, 7.41, -1.48],
        [4180, -1.385, 7.09, -1.36],
        [4308, -1.380, 7.08, -1.37],
        [4436, -1.362, 6.95, -1.35],
        [4564, -1.365, 6.92, -1.31],
        [4692, -1.339, 6.56, -1.17],
        [4820, -1.371, 7.06, -1.40],
        [4948, -1.358, 6.91, -1.34],
        [5052, -1.360, 6.91, -1.33],
        [5180, -1.353, 6.74, -1.25],
        [5308, -1.359, 6.82, -1.27],
        [5436, -1.380, 7.05, -1.37],
        [5564, -1.376, 6.99, -1.31],
        [5692, -1.405, 7.39, -1.47],
        [5820, -1.394, 7.29, -1.45],
        [5948, -1.428, 7.57, -1.57],
        [6052, -1.445, 7.68, -1.50],
        [6148, -1.422, 7.38, -1.38],
        [6308, -1.463, 7.94, -1.62],
        [6436, -1.478, 8.22, -1.74],
        [6564, -1.473, 8.00, -1.62],
        [6692, -1.455, 7.76, -1.53],
        [6820, -1.487, 8.22, -1.72],
        [6948, -1.472, 8.05, -1.67],
        [7052, -1.470, 8.01, -1.64],
        [7180, -1.503, 8.50, -1.84],
        [7308, -1.482, 8.19, -1.72],
        [7436, -1.498, 8.22, -1.66],
        [7564, -1.490, 8.18, -1.66],
        [7692, -1.481, 7.98, -1.56],
        [7820, -1.474, 7.94, -1.57],
        [7948, -1.448, 7.69, -1.51],
        [8308, -1.402, 7.16, -1.35],
        [8436, -1.400, 7.12, -1.32],
        [13308, -1.403, 7.37, -1.47],
        [13436, -1.392, 7.08, -1.31],
        [13564, -1.384, 6.94, -1.24],
        [13692, -1.382, 6.95, -1.25],
        [13820, -1.376, 6.88, -1.24],
        [13948, -1.384, 6.98, -1.28],
        [14052, -1.400, 7.36, -1.48],
        [14180, -1.397, 7.29, -1.45]
    ]).T

    ind = np.argsort(coeffs[0])
    coeffs = coeffs[:, ind]

    freqs = coeffs[0]
    poly = coeffs[1:].T
    assert freq <= freqs.max()
    assert freq >= freqs.min()

    ind = np.searchsorted(freqs, freq)
    flower = freqs[ind - 1]
    clower = poly[ind - 1]
    fupper = freqs[ind]
    cupper = poly[ind]
    rweight = (freq - flower)/(fupper - flower)

    xf, xp = dom.distances[0]*dom.shape[0]/2, dom.shape[0]
    yf, yp = dom.distances[1]*dom.shape[1]/2, dom.shape[1]
    xx, yy = np.meshgrid(
        np.linspace(-xf, xf, xp), np.linspace(-yf, yf, yp), indexing='ij')
    r = np.sqrt(xx**2 + yy**2)/1000/ARCMIN2RAD  # Mhz->GHz, RAD->ARCMIN
    lower = _vla_eval_poly(clower, flower*r)
    upper = _vla_eval_poly(cupper, fupper*r)
    beam = rweight*upper + (1 - rweight)*lower
    beam[beam < 0] = 0
    return ift.from_global_data(dom, beam)

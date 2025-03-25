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

import pickle
from time import time

import numpy as np

import nifty5 as ift


def print_section(s):
    print(80*'-')
    print(' {}'.format(s))
    print(80*'-')


def power_analyze(field):
    dom = field.domain[0]
    ht = ift.HartleyOperator(dom.get_default_codomain(), dom)
    return ift.power_analyze(ht.inverse(field))


def field2fits(field, file_name):
    from astropy.io import fits
    hdu = fits.PrimaryHDU()
    hdu.data = field.to_global_data().T
    hdu.writeto(file_name, overwrite=True)


def fits2field(file_name):
    from astropy.io import fits
    hdu_list = fits.open(file_name)
    # FIXME Write proper shapes
    shp = hdu_list[0].data.shape
    if len(shp) == 2:
        image_data = hdu_list[0].data.T
    elif len(shp) == 3:
        image_data = hdu_list[0].data[0].T
    elif len(shp) == 4:
        image_data = hdu_list[0].data[0, 0].T
    else:
        raise NotImplementedError
    dstx = abs(hdu_list[0].header['CDELT1']*np.pi/180)
    dsty = abs(hdu_list[0].header['CDELT2']*np.pi/180)
    nx = int(abs(hdu_list[0].header['NAXIS1']))
    ny = int(abs(hdu_list[0].header['NAXIS2']))
    dom = ift.RGSpace((nx, ny), (dstx, dsty))
    return ift.from_global_data(dom, image_data)


def calibrator_sky(domain, flux):
    middle_coords = tuple([i//2 for i in domain.shape])
    sky_c = np.zeros(domain.shape)
    sky_c[middle_coords] = flux
    return ift.Field.from_global_data(domain, sky_c).weight(-1)


def save_pickle(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(fname):
    try:
        with open(fname, 'rb') as f:
            pos = pickle.load(f)
        return pos
    except FileNotFoundError:
        raise FileNotFoundError('Input file not found.')
    except OSError:
        print(
            'Something went wrong during loading and unpickling of input file.'
        )
        exit()


def zero_to_nan(fld):
    dom = fld.domain
    fld = fld.to_global_data_rw()
    fld[fld == 0] = np.nan
    return ift.from_global_data(dom, fld)


def tuple_to_image(fld):
    dom = fld.domain
    assert len(dom.shape) == 2 and isinstance(
        dom[0], ift.UnstructuredDomain) and isinstance(dom[1], ift.RGSpace)
    vol = dom[1].shape[0]*dom[1].distances[0]
    newdom = ift.RGSpace(
        dom.shape[::-1], distances=[dom[1].distances[0], vol/dom.shape[0]/2])
    newdom = ift.RGSpace(dom.shape[::-1], distances=[dom[1].distances[0], 1.])
    return ift.from_global_data(newdom, fld.to_global_data().T)


def print_kl_sample_statistics(kl, ham):
    sc = ift.StatCalculator()
    for samp in kl.samples:
        sc.add(ham(kl.position.flexible_addsub(samp, False)))
    print('Hamiltonian = {:.2E} +/- {:.2E}'.format(
        float(sc.mean.to_global_data()),
        float(ift.sqrt(sc.var).to_global_data())))


def time_operator(op, n=3, want_metric=True):
    pos = ift.from_random('normal', op.domain)
    t0 = time()
    for _ in range(n):
        res = op(pos)
    print('Operator call with field:', (time() - t0)/n)

    lin = ift.Linearization.make_var(pos, want_metric=want_metric)
    t0 = time()
    for _ in range(n):
        res = op(lin)
    print('Operator call with linearization:', (time() - t0)/n)

    lin = ift.Linearization.make_var(pos, want_metric=want_metric)
    t0 = time()
    for _ in range(n):
        res.gradient
    print('Gradient evaluation:', (time() - t0)/n)

    t0 = time()
    for _ in range(n):
        res.metric(pos)
    print('Metric apply:', (time() - t0)/n)


def points2mask(arr, space):
    arr = np.array(arr)
    assert len(space) == 1
    space = ift.DomainTuple.make(space)[0]
    assert arr.shape[1] == 4
    res = np.zeros(space.shape)
    for ii in range(arr.shape[0]):
        midx, midy = [ss//2 for ss in space.shape]
        x0, y0, x1, y1 = arr[ii]
        x0 = int(np.round(x0/space.distances[0]))
        x1 = int(np.round(x1/space.distances[0]))
        y0 = int(np.round(y0/space.distances[1]))
        y1 = int(np.round(y1/space.distances[1]))
        res[midx + x0:midx + x1 + 1, midy + y0:midy + y1 + 1] = 1
    return ift.from_global_data(space, res)

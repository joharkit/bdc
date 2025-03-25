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
from scipy.stats import norm

import nifty5 as ift


class UniformOperator(ift.Operator):
    def __init__(self, domain, slc, lower, upper):
        self._domain = self._target = ift.DomainTuple.make(domain)
        self._slc = tuple(slc)
        if len(slc) > len(self._domain.shape):
            raise ValueError
        lower, upper = float(lower), float(upper)
        if not lower < upper:
            raise ValueError
        self._scale, self._offset = upper - lower, lower

    def apply(self, x):
        self._check_input(x)

        lin = isinstance(x, ift.Linearization)
        val = x.val.to_global_data() if lin else x.to_global_data()

        res = norm.cdf(val[self._slc])
        if self._scale != 1.:
            res *= self._scale
        if self._offset != 0.:
            res += self._offset

        resfld = val.copy()
        resfld[self._slc] = res
        resfld = ift.Field.from_local_data(self._domain, resfld)
        if not lin:
            return resfld

        derfld = np.ones(self._domain.shape)
        derfld[self._slc] = self._scale*norm.pdf(val[self._slc])
        jac = ift.makeOp(ift.Field.from_global_data(self._domain, derfld))
        jac = jac(x.jac)
        return x.new(resfld, jac)

    def pre_image(self, x):
        res = x.to_global_data_rw()
        a = (res[self._slc] - self._offset)/self._scale
        o = norm.ppf(a)
        if np.isnan(o):
            s = 'Allowed lower and upper bound for zeromode: '
            s += '{} -- {}\n'.format(self._offset, self._offset + self._scale)
            s += 'Actual value: {}'.format(a)
            raise ValueError(s)
        res[self._slc] = o
        return ift.Field.from_global_data(x.domain, res)

    @property
    def upper(self):
        return self._scale + self._offset

    @property
    def lower(self):
        return self._offset

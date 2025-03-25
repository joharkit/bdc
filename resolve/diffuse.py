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

from .extended_operator import AXiOperator
from .sugar import power_analyze


class Diffuse(AXiOperator):
    def __init__(self, target, amplitude, mean, stddev, xi_key):
        target = ift.DomainTuple.make(target)
        if len(target) != 1:
            raise ValueError
        if len(target.shape) != 2:
            raise ValueError
        h_space = target[0].get_default_codomain()
        self._ht = ift.HartleyOperator(h_space, target[0])
        p_space = amplitude.target[0]
        power_distributor = ift.PowerDistributor(h_space, p_space)

        self._amplitude = amplitude
        self._A = power_distributor @ self._amplitude
        self._xi = ift.ducktape(h_space, None, xi_key)

        dom = self._xi.target
        zm_sd = np.ones(dom.shape)
        zm_sd[0, 0] = stddev
        self._zmsd = ift.from_global_data(dom, zm_sd)
        zm_sd = ift.makeOp(self._zmsd)
        zm_mean = np.zeros(dom.shape)
        zm_mean[0, 0] = mean
        self._zmm = ift.from_global_data(dom, zm_mean)
        zm_mean = ift.Adder(self._zmm)

        self._logop = (self._ht @ zm_mean @ zm_sd)(self._xi*self._A)
        self.xionly = (self._ht @ zm_mean @ zm_sd)(self._xi)
        self._op = self._logop.clip(-50, 50).exp()

        # Checks
        fld = ift.from_random('normal', self._A.domain)
        assert self._A(fld).to_global_data()[0, 0] == 1.

    @property
    def pspec(self):
        return self._amplitude**2

    def pre_image(self, field):
        f = (self._ht.inverse(field.log()) - self._zmm)/self._zmsd
        pos = ift.full(self._A.domain, 0.)
        A = self._A(pos)
        xi = ift.MultiField.from_dict({self._xi.domain.keys()[0]: f/A})
        return ift.MultiField.union([pos, xi])

    @property
    def log_op(self):
        return self._logop

    @classmethod
    def model2params(cls, skymodel, sm):
        logskymodel = skymodel.log()
        pspec = power_analyze(logskymodel)
        t0 = np.log(pspec.domain[0].k_lengths[1])
        im = np.log(pspec.to_global_data()[1]) - t0*float(sm)
        zm = logskymodel.integrate()
        zmstd = (logskymodel*0 + 2).integrate()
        return {'im': im, 'zm': zm, 'zmstd': zmstd}


def default_pspace(space):
    sp = ift.DomainTuple.make(space)
    if not len(sp) == 1:
        raise ValueError
    sp = sp[0].get_default_codomain()
    return ift.PowerSpace(sp)

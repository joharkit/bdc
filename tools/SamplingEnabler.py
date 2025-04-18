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
# Copyright(C) 2013-2020 Max-Planck-Society

import numpy as np
import nifty5 as ift

class SamplingEnabler(ift.EndomorphicOperator):
    """Class which acts as a operator object built of (`likelihood` + `prior`)
    and enables sampling from its inverse even if the operator object
    itself does not support it.


    Parameters
    ----------
    likelihood : :class:`EndomorphicOperator`
        Metric of the likelihood
    prior : :class:`EndomorphicOperator`
        Metric of the prior
    iteration_controller : :class:`IterationController`
        The iteration controller to use for the iterative numerical inversion
        done by a :class:`ConjugateGradient` object.
    approximation : :class:`LinearOperator`, optional
        if not None, this linear operator should be an approximation to the
        operator, which supports the operation modes that the operator doesn't
        have. It is used as a preconditioner during the iterative inversion,
        to accelerate convergence.
    start_from_zero : boolean
        If true, the conjugate gradient algorithm starts from a field filled
        with zeros. Otherwise, it starts from a prior samples. Default is
        False.
    """

    def __init__(self, likelihood, iteration_controller,
                 approximation=None):
        self._likelihood = likelihood
        self._ic = iteration_controller
        self._approximation = approximation
        self._start_from_zero = True
        self._op = likelihood
        self._domain = self._op.domain
        self._capability = self._op.capability

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        try:
            return self._op.draw_sample(from_inverse, dtype)
        except NotImplementedError:
            if not from_inverse:
                raise ValueError("from_inverse must be True here")
            if self._start_from_zero:
                b = self._op.draw_sample()
                energy = ift.QuadraticEnergy(0*b, self._op, b)
            else:
                s = self._prior.draw_sample(from_inverse=True)
                sp = self._prior(s)
                nj = self._likelihood.draw_sample()
                energy = ift.QuadraticEnergy(s, self._op, sp + nj,
                                         _grad=self._likelihood(s) - nj)
            inverter = ift.ConjugateGradient(self._ic)
            if self._approximation is not None:
                energy, convergence = inverter(
                    energy, preconditioner=self._approximation.inverse)
            else:
                energy, convergence = inverter(energy)
            return energy.position

    def apply(self, x, mode):
        return self._op.apply(x, mode)

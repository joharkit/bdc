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

import nifty5 as ift


class ExtendedOperator(ift.Operator):
    @property
    def _domain(self):
        return self._op.domain

    @property
    def _target(self):
        return self._op.target

    def apply(self, x):
        self._check_input(x)
        return self._op(x)

    def __repr__(self):
        return self._op.__repr__()


class AXiOperator(ExtendedOperator):
    def adjust_variances(self, position, samples=[]):
        # Check whether necessary stuff is defined
        self._A, self._xi, self._op

        ham = ift.make_adjust_variances_hamiltonian(
            self._A, self._xi, position, samples=samples)
        e = ift.EnergyAdapter(
            position.extract(ham.domain), ham, constants=[], want_metric=True)
        ic = ift.GradInfNormController(
            1e-8,
            convergence_level=3,
            iteration_limit=200,
            name='Adjust variances')
        minimizer = ift.NewtonCG(ic)
        e, _ = minimizer(e)
        s_h_old = (self._A*self._xi).force(position)
        position = e.position.to_dict()
        position[list(self._xi.domain.keys())[0]] = s_h_old/self._A(e.position)
        return ift.MultiField.from_dict(position)

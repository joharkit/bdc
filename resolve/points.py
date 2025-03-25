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
from nifty5.utilities import my_sum

from .extended_operator import ExtendedOperator


class Points(ExtendedOperator):
    def __init__(self, target, alpha, q, key=None):
        self._op = ift.InverseGammaOperator(target, alpha=alpha, q=q)
        if key is not None:
            self._op = self._op.ducktape(key)
        self._alpha = float(alpha)
        self._q = float(q)
        self._key = key

    def pre_image(self, field):
        if not isinstance(field, ift.Field):
            raise TypeError
        fld = ift.InverseGammaOperator.inverseIG(
            field, alpha=self._alpha, q=self._q)
        if self._key is not None:
            return ift.MultiField.from_dict({self._key: fld})
        return fld


# FIXME Simplify starblade
def separate_point_sources(sky,
                           alpha,
                           q,
                           manual_power_spectrum=None,
                           initial_power_value=1e-4):
    s_space = sky.domain[0]
    h_space = s_space.get_default_codomain()
    FFT = ift.HartleyOperator(h_space, target=s_space)
    binbounds = ift.PowerSpace.useful_binbounds(h_space, logarithmic=False)
    p_space = ift.PowerSpace(h_space, binbounds=binbounds)
    if manual_power_spectrum is None:
        initial_spectrum = ift.Field.full(p_space, float(initial_power_value))
        update_power = True
    else:
        initial_spectrum = manual_power_spectrum
        update_power = False
    initial_x = ift.Field.full(s_space, -1.)
    alpha = ift.Field.full(s_space, alpha)
    q = ift.Field.full(s_space, q)
    ICI = ift.GradientNormController(iteration_limit=10, tol_abs_gradnorm=1e-5)
    IC_samples = ift.GradientNormController(
        iteration_limit=1000, tol_abs_gradnorm=1e-5)
    parameters = dict(
        data=sky,
        power_spectrum=initial_spectrum,
        alpha=alpha,
        q=q,
        controller=ICI,
        FFT=FFT,
        sampling_controller=IC_samples,
        update_power=update_power)
    starblade = StarbladeEnergy(position=initial_x, parameters=parameters)
    for i in range(2):
        starblade = starblade_iteration(
            starblade,
            samples=5,
            cg_steps=10,
            newton_steps=100,
            sampling_steps=1000)
    points = starblade.point_like
    diffuse = starblade.diffuse
    assert points.domain is sky.domain
    assert diffuse.domain is sky.domain
    return points, diffuse


def starblade_iteration(starblade,
                        samples=5,
                        cg_steps=10,
                        newton_steps=100,
                        sampling_steps=1000):
    """Performs one Newton minimization step

    Parameters
    ----------
    starblade : StarbladeEnergy
        An instance of an Starblade Energy
    samples : int
        Number of samples drawn in order to estimate the KL. If zero the MAP
        is calculated (default: 5).
    cg_steps : int
        Maximum number of conjugate gradient iterations for
        numerical operator inversion for each Newton step (default: 10).
    newton_steps : int
        Number of consecutive Newton steps within one algorithmic step
        (default: 100)
    sampling_steps : int
        Number of conjugate gradient steps for each sample (default: 1000).
    """
    controller = ift.GradientNormController(
        name="Newton", tol_abs_gradnorm=1e-8, iteration_limit=newton_steps)
    minimizer = ift.RelaxedNewton(controller=controller)
    ICI = ift.GradientNormController(
        iteration_limit=cg_steps, tol_abs_gradnorm=1e-5)
    IC_samples = ift.GradientNormController(
        iteration_limit=sampling_steps, tol_abs_gradnorm=1e-5)
    para = dict(starblade.parameters)
    para['controller'] = ICI
    para['sampling_controller'] = IC_samples
    energy = StarbladeEnergy(starblade.position, parameters=para)
    if samples > 0:
        sample_list = [
            energy.metric.inverse.draw_sample() for _ in range(samples)
        ]
        energy = StarbladeKL(energy, sample_list)
    else:
        energy = starblade
    energy, convergence = minimizer(energy)
    energy = StarbladeEnergy(energy.position, parameters=energy.parameters)
    sample_list = [energy.metric.inverse.draw_sample() for _ in range(samples)]
    if len(sample_list) == 0:
        sample_list.append(energy.position)
    new_position = energy.position
    new_parameters = dict(energy.parameters)
    if energy.parameters['update_power']:
        new_parameters['power_spectrum'] = update_power(energy)
    return StarbladeEnergy(new_position, new_parameters)


def update_power(energy):
    """Calculates a new estimate of the power spectrum given a StarbladeEnergy
    or StarbladeKL. For Energy the MAP estimate of the power spectrum is
    calculated and for KL the variational estimate.

    Parameters
    ----------
    energy : StarbladeEnergy or StarbladeKL
        An instance of an StarbladeEnergy or StarbladeKL
    """

    def _analyze(en):
        return ift.power_analyze(
            energy.parameters['FFT'].inverse(en.s),
            binbounds=en.parameters['power_spectrum'].domain[0].binbounds)

    if isinstance(energy, StarbladeKL):
        return ift.my_sum(map(_analyze, energy._energy_list))/len(
            energy._energy_list)
    else:
        return _analyze(energy)


class SampledKullbachLeiblerDivergence(ift.Energy):
    def __init__(self, h, res_samples):
        super(SampledKullbachLeiblerDivergence, self).__init__(h.position)
        self._h = h
        self._res_samples = res_samples

        self._energy_list = tuple(
            h.at(self.position + ss) for ss in res_samples)

    def at(self, position):
        return self.__class__(self._h.at(position), self._res_samples)

    @property
    @ift.memo
    def value(self):
        return (my_sum(map(lambda v: v.value, self._energy_list))/len(
            self._energy_list))

    @property
    @ift.memo
    def gradient(self):
        return (my_sum(map(lambda v: v.gradient, self._energy_list))/len(
            self._energy_list))

    @property
    @ift.memo
    def metric(self):
        return (my_sum(map(lambda v: v.metric, self._energy_list)).scale(
            1./len(self._energy_list)))


class StarbladeKL(SampledKullbachLeiblerDivergence):
    """The Kullback-Leibler divergence for the starblade problem.

    Parameters
    ----------
    energy : StarbladeEnergy
        The energy to sample from
    samples : List
        A list containing residual samples.
    """

    def __init__(self, energy, samples):
        super(StarbladeKL, self).__init__(energy, samples)

    @property
    def parameters(self):
        return self._energy_list[0].parameters

    @property
    def metric(self):
        metric = SampledKullbachLeiblerDivergence.metric.fget(self)
        return ift.InversionEnabler(metric, self.parameters['controller'])


class StarbladeEnergy(ift.Energy):
    """The Energy for the starblade problem.

    Implements the Information Hamiltonian of the separation of d.

    Parameters
    ----------
    position : Field
        The current position of the separation.
    parameters : Dictionary
        Dictionary containing all relevant quantities for the inference,
        data : Field
            The image data.
        alpha : Field
            Slope parameter of the point-source prior
        q : Field
            Cutoff parameter of the point-source prior
        power_spectrum : callable or Field
            An object that contains the power spectrum of the diffuse component
             as a function of the harmonic mode.
        FFT : HartleyOperator
            An operator performing the Hartley transform
        controller : IterationController
            The minimization strategy to use for operator inversion
        sampling_controller :
            Iteration controller which is used to generate samples.
    """

    def __init__(self, position, parameters):
        x = position.local_data.clip(-9, 9)
        position = ift.Field.from_local_data(position.domain, x)
        super(StarbladeEnergy, self).__init__(position=position)

        self.parameters = ift.frozendict(parameters)
        self.FFT = parameters['FFT']
        self.correlation = ift.create_power_operator(
            self.FFT.domain, parameters['power_spectrum'])
        self.alpham1 = parameters['alpha'] - 1.
        self.S = ift.SandwichOperator.make(self.FFT.adjoint, self.correlation)
        tmp = ift.tanh(position)
        self.a = 0.5*(1. + tmp)
        self.a_p = 0.5*(1. - tmp**2)
        self.a_pp = 2. - 4.*self.a
        da = parameters['data']*self.a
        self.u = ift.log(da)
        self.qexpmu = parameters['q']/da
        self.u_p = self.a_p/self.a
        self.u_a_sum = -ift.log(self.a).sum()
        one_m_a = 1. - self.a
        self.s = ift.log(parameters['data']*one_m_a)
        self.s_p = -self.a_p/one_m_a
        self.var_x = 9.

    def at(self, position):
        return self.__class__(position, parameters=self.parameters)

    @property
    def diffuse(self):
        return ift.exp(self.s)

    @property
    def point_like(self):
        return ift.exp(self.u)

    @property
    def value(self):
        diffuse = 0.5*self.s.vdot(self.S.inverse_times(self.s))
        point = self.alpham1.vdot(self.u) + self.qexpmu.sum()
        det = -self.s.sum() - self.u_a_sum - ift.log(self.a_p).sum()
        det += 0.5/self.var_x*self.position.vdot(self.position)
        return diffuse + point + det

    @property
    def gradient(self):
        diffuse = self.S.inverse_times(self.s)*self.s_p
        point = self.u_p*(self.alpham1 - self.qexpmu)
        det = (self.position/self.var_x - self.s_p + self.u_p - self.a_pp)
        return diffuse + point + det

    @property
    def metric(self):
        point = self.qexpmu*self.u_p**2

        O_x = ift.ScalingOperator(1./self.var_x, self.position.domain)
        N_inv = ift.DiagonalOperator(point)
        S_p = ift.DiagonalOperator(self.s_p)
        my_S_inv = ift.SandwichOperator.make(
            self.FFT.inverse(S_p), self.correlation.inverse)
        curv = ift.InversionEnabler(
            ift.SamplingEnabler(my_S_inv + N_inv, O_x,
                                self.parameters['sampling_controller']),
            self.parameters['controller'])
        return curv

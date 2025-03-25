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
from operator import add
from time import time

import numpy as np
from matplotlib.colors import LogNorm

import nifty5 as ift

from .amplitude_operators import SLAmplitude
from .diffuse import Diffuse, default_pspace
from .likelihood import LikelihoodMaker
from .points import Points, separate_point_sources
from .sugar import (calibrator_sky, load_pickle, power_analyze, save_pickle,
                    tuple_to_image)
from .version import gitversion


def _calc_mean_range(op, N):
    sc = ift.StatCalculator()
    for _ in range(N):
        fld = op(ift.from_random('normal', op.domain))
        arr = fld.to_global_data()
        sc.add(np.max(arr) - np.min(arr))
    return sc.mean


class Imager:
    def __init__(self, newton_steps, target_data_handler, file_prefix=''):
        # Create output directory
        self._prefix = str(file_prefix)

        # Save version information
        ss = 'resolve: {}\n'.format(gitversion())
        ss += 'NIFTy version: {}'.format(ift.version.gitversion())
        print(ss)

        self.dh_t = target_data_handler
        self._diffuse = self._points = None
        self._position = None

        self._counter = 0
        ic = ift.AbsDeltaEnergyController(
            0.5,
            convergence_level=5,
            iteration_limit=int(newton_steps),
            name='NewtonCG',
            file_name='{}energy_newton.log'.format(self._prefix))
        self._minimizer = ift.NewtonCG(
            ic, file_name='{}energy_inverter.log'.format(self._prefix))
        self.cal_ops = {}
        self._alpha = self._q = 1

    def init_points(self, alpha, q, mask=None):
        sky_space = self.dh_t.sky_domain[0]
        self._points = Points(sky_space, alpha=alpha, q=q, key='points')
        if mask is not None:
            self._points = ift.makeOp(mask) @ self._points
        self._update_position_domain()

    def init_diffuse(self, a, k0, sm, sv, im, iv, zm, zmstd, mask=None):
        sky_space = self.dh_t.sky_domain[0]
        a = SLAmplitude(target=default_pspace(sky_space),
                        n_pix=64,
                        a=a,
                        k0=k0,
                        sm=sm,
                        sv=sv,
                        im=im,
                        iv=iv,
                        keys=['diffuse tau', 'diffuse phi'])
        self._diffuse = Diffuse(sky_space, a, zm, zmstd, 'diffuse xi')
        if mask is not None:
            self._diffuse = ift.makeOp(mask) @ self._diffuse
        self._update_position_domain()

    def find_parameters(self, sm, left, right, tol, alpha=0.5):
        func = lambda x: self._golden_helper(float(alpha), x)
        left, right, tol = np.log(left), np.log(right), np.log(tol)

        # Golden section search
        phi = 2/(np.sqrt(5) - 1)
        left, right = min(left, right), max(left, right)
        width = right - left
        if width <= tol:
            res = (left, right)
        else:
            n = int(np.ceil(np.log(tol/width)/np.log(1/phi))) - 1
            print('Will need {} minimizations.'.format(3 + n))
            x0 = left + width/phi**2
            x1 = left + width/phi
            s0, y0 = func(x0)
            s1, y1 = func(x1)
            for k in range(n):
                print('Left: {}, right: {}'.format(np.exp(left),
                                                   np.exp(right)))
                width /= phi
                if y0 < y1:
                    right = x1
                    x1, y1 = x0, y0
                    x0 = left + width/phi**2
                    s0, y0 = func(x0)
                else:
                    left = x0
                    x0, y0 = x1, y1
                    x1 = left + width/phi
                    s1, y1 = func(x1)
            if y0 < y1:
                res = (left, x1)
            else:
                res = (x0, right)
        # End golden section search

        q = np.exp(res[0] + 0.5*(res[1] - res[0]))
        print('Found q = {}'.format(q))
        sky, _ = self._golden_helper(alpha, np.log(q))
        self.dh_t.writefits(sky, '{}find_parameters.fits'.format(self._prefix))
        params = Diffuse.model2params(sky, sm)
        im = params['im']
        zm = params['zm']
        zmstd = params['zmstd']
        return sm, im, zm, zmstd, q, alpha, sky

    def _golden_helper(self, alpha, logq, iterations=10):
        sky, e = self._ig_imaging(alpha, np.exp(logq), int(iterations))

        p = ift.Plot()
        p.add(sky,
              norm=LogNorm(),
              title='q={:.2E}, integral log flux {:.2E}'.format(
                  np.exp(logq),
                  sky.log().integrate()))
        p.add(power_analyze(sky.log()), label='Power analyze')

        residual = ift.full(sky.domain, 0.)
        rs, invvars, viss = self.dh_t.get_Rinvvarvis_iter()
        rs, invvars, viss = list(rs), list(invvars), list(viss)
        for ii in range(len(rs)):
            residual = residual + rs[ii].adjoint(
                (rs[ii](sky) - viss[ii])*invvars[ii])
        p.add(residual, title='Residual image')

        fname = '{}cgimaging.'.format(self._prefix)
        p.output(name=fname + 'png', ny=1, xsize=20, ysize=8)
        return sky, e

    def ig_imaging(self, alpha, q, iterations=20, initial_sky=None):
        sky, e = self._ig_imaging(alpha,
                                  q,
                                  iterations=iterations,
                                  initial_sky=initial_sky)
        return sky

    def _ig_imaging(self, alpha, q, iterations=20, initial_sky=None):
        sky = Points(self.dh_t.sky_domain, alpha, q)
        rs, invvars, viss = self.dh_t.get_Rinvvarvis_iter()
        e = []
        for r, vis, invvar in zip(rs, viss, invvars):
            e.append(
                ift.GaussianEnergy(inverse_covariance=ift.makeOp(invvar),
                                   mean=vis) @ r)
        lh = reduce(add, e) @ sky

        if initial_sky is None:
            pos = ift.full(lh.domain, 0)
        else:
            pos = sky.pre_image(initial_sky)
        e = ift.EnergyAdapter(pos,
                              ift.StandardHamiltonian(lh),
                              want_metric=True)
        mini = ift.NewtonCG(
            ift.GradInfNormController(1e-7,
                                      convergence_level=3,
                                      iteration_limit=iterations))
        e, _ = mini(e)
        return sky(e.position), e.value

    def kl(self,
           n_samples,
           point_estimates=[],
           constants=[],
           napprox=0,
           sampling_iteration_limit=2000):
        lh = self.lhm_t.get_full()
        ic_samp = ift.AbsDeltaEnergyController(
            1,
            convergence_level=3,
            iteration_limit=sampling_iteration_limit,
            name='Sampling',
            file_name='{}energy_sampling.log'.format(self._prefix))

        # Set constant directions
        pe, cst = [], []
        keysdct = {
            'diffuse': ['diffuse xi', 'diffuse phi', 'diffuse tau'],
            'points': ['points'],
            'diffuse power spectrum': ['diffuse phi', 'diffuse tau']
        }
        for kk in point_estimates:
            pe = [*pe, *keysdct[kk]]
        for kk in constants:
            cst = [*cst, *keysdct[kk]]
        # End set constant directions

        e = ift.MetricGaussianKL(self.position.extract(lh.domain),
                                 ift.StandardHamiltonian(lh, ic_samp),
                                 n_samples,
                                 napprox=napprox,
                                 point_estimates=pe,
                                 mirror_samples=True,
                                 constants=cst)
        self.plot_samples([e.position + samp for samp in e.samples],
                          'samples{}'.format(int(time())))
        return e

    def hamiltonian(self, constants=[]):
        cst = []
        lh = self.lhm_t.get_full()
        if 'diffuse' in constants:
            cst.append('diffuse xi')
            cst.append('diffuse tau')
            cst.append('diffuse phi')
        if 'points' in constants:
            cst.append('points')
        return ift.EnergyAdapter(self._position,
                                 ift.StandardHamiltonian(lh),
                                 constants=cst,
                                 want_metric=True)

    def minimize(self, e):
        e, _ = self._minimizer(e)
        a = self.dh_t.ndof/2
        b = self.position.size/2
        print('0.5*Data points: {}'.format(a))
        print('0.5*Minimization degrees of freedom: {}'.format(b))
        print('Relative value of energy: {}'.format(e.value/(a + b)))
        self.update_position(e.position)

    def starblade(self, alpha, q):
        # FIXME Add option to use the current power spectrum. Non-trivial!
        sky = (self._diffuse + self._points).force(self._position)
        p, d = separate_point_sources(sky, alpha, q)
        diff = self._diffuse.pre_image(d)
        pointlike = self._points.pre_image(p)
        pos = ift.MultiField.union([pointlike, diff])
        self.update_position(pos)

    def adjust_variances(self, n_samples):
        samples = []
        if n_samples > 0:
            samples = self.kl(n_samples).samples
        pos = self._diffuse.adjust_variances(self.position, samples)
        self.update_position(pos)

    ###########################################################################
    # Plotting and saving
    ###########################################################################

    def save(self, name='', position=None):
        pos = self.position if position is None else position
        self.plot_overview('{:02}{}'.format(self._counter, name),
                           position=position)
        s = '{}{:02}{}'.format(self._prefix, self._counter, name)
        save_pickle(pos, s + '.pickle')
        self.dh_t.writefits(self.sky.force(pos), s + '.fits')
        if 'points' in self.position:
            self.dh_t.writefits(self.points.force(pos), s + '_points.fits')
        if 'diffuse xi' in self.position:
            self.dh_t.writefits(self.diffuse.force(pos), s + '_diffuse.fits')
        self._counter += 1

    def plot_samples(self, samples, name):
        p = ift.Plot()
        sc = ift.StatCalculator()
        for samp in samples:
            ss = self.sky.force(samp)
            integral = ss.log().integrate()
            p.add(ss, norm=LogNorm(), title='Integral {:.2E}'.format(integral))
            sc.add(ss)
        p.add(sc.mean, norm=LogNorm(), title='Mean')
        p.add(sc.var, norm=LogNorm(), title='Variance')
        p.add(sc.var.sqrt()/sc.mean, title='Std dev/mean')
        p.add(sc.var.sqrt()/sc.mean, norm=LogNorm(), title='Std dev/mean')
        if self.diffuse is not None:
            pspecs = []
            for samp in samples:
                pspecs.append(self._diffuse.pspec.force(samp))
            p.add(pspecs)
        p.output(name='{}{}.png'.format(self._prefix, name),
                 xsize=20,
                 ysize=20)

    def plot_prior_samples(self, n=15):
        self.plot_samples([
            ift.from_random('normal', self.position.domain)
            for _ in range(int(n))
        ], 'prior_samples')

    def plot_overview(self, name, position=None):
        if position is None:
            position = self.position
        p = ift.Plot()
        dct = {'colormap': 'inferno'}
        sky = self.sky.force(position)
        p.add(sky, title='Sky', **dct)
        p.add(sky, title='Sky', norm=LogNorm(), **dct)
        if 'diffuse xi' in self.position:
            diffuse = self._diffuse.force(position)
            p.add(diffuse,
                  norm=LogNorm(),
                  title='Diffuse (Zero mode excitation: {:.2E})'.format(
                      position['diffuse xi'].to_global_data()[0, 0]),
                  **dct)
            p.add(self._diffuse.xionly.force(position),
                  title='ht(xi) without exp!',
                  **dct)
            flds = {'Reconstruction': self._diffuse.pspec.force(position)}
            p.add(flds.values(),
                  label=list(flds.keys()),
                  title='Power spectrum, integral {:.2E}'.format(
                      diffuse.clip(1e7, None).log().integrate()),
                  xlabel='Spatial frequency [1/rad]',
                  **dct)
        if 'points' in self.position:
            p.add(self._points.force(position),
                  norm=LogNorm(),
                  title='Points',
                  **dct)
        resj = self.lhm_t.residual_image(position)
        zmax = np.max(np.abs(resj.to_global_data()))
        p.add(resj,
              title='Residual information source t',
              zmin=-zmax,
              zmax=zmax)
        self.dh_t.writefits(resj,
                            '{}{}_residual.fits'.format(self._prefix, name))
        p.output(name='{}{}.png'.format(self._prefix, name),
                 nx=3,
                 xsize=25,
                 ysize=25)

    ###########################################################################
    # Convience functions for position
    ###########################################################################

    def update_position(self, position):
        self.position = ift.MultiField.union([self.position, position])

    def _update_position_domain(self):
        doms = [ll.get_full().domain for ll in self.likelihoods.values()]
        dom = ift.MultiDomain.union(doms)
        if self._position is None:
            self._position = 0.1*ift.from_random('normal', dom)
        else:
            self._position = self.position.unite(
                0.1*ift.from_random('normal', dom))

    def load_position(self, name, counter=None):
        self.position = load_pickle(name)
        if counter is not None:
            self._counter = int(counter)

    ###########################################################################
    # Getters and setters
    ###########################################################################

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if value.domain is not self.position.domain:
            ss = 'New value domain:\n' + str(value.domain) + '\n'
            ss += 'Old value domain:\n' + str(self.position.domain)
            raise ValueError(ss)
        self._position = value
        for kk, vv in value.items():
            vv = vv.to_global_data()
            m, sd = np.mean(vv), np.std(vv)
            print('{}: {} +/- {}'.format(kk, m, sd))

    @property
    def lhm_t(self):
        return LikelihoodMaker(self.dh_t,
                               self.sky,
                               self.cal_ops,
                               alpha=self._alpha,
                               q=self._q)

    @property
    def diffuse(self):
        return self._diffuse

    @property
    def points(self):
        return self._points

    @property
    def sky(self):
        foo = []
        if self._points is not None:
            foo.append(self._points)
        if self.diffuse is not None:
            foo.append(self._diffuse)
        return reduce(add, foo)

    @property
    def data_handlers(self):
        a = {}
        if hasattr(self, 'dh_t'):
            a['t'] = self.dh_t
        if hasattr(self, 'dh_c'):
            a['c'] = self.dh_c
        return a

    @property
    def likelihoods(self):
        a = {}
        if hasattr(self, 'lhm_t'):
            a['t'] = self.lhm_t
        if hasattr(self, 'lhm_c'):
            a['c'] = self.lhm_c
        return a

    @property
    def _isset_target(self):
        return hasattr(self, 'lhm_t')

    @property
    def _isset_calibrator(self):
        return hasattr(self, 'lhm_c')

    @property
    def out(self):
        return self._out


class JointImagerCalibration(Imager):
    def get_full_kl(self, n_samples, cg_samp):
        lh = self.lhm_t.get_full()
        if self._isset_calibrator:
            lh = lh + self.lhm_c.get_full()
        ic_samp = ift.GradientNormController(iteration_limit=cg_samp)
        dct = {
            'mean': self.position,
            'hamiltonian': ift.StandardHamiltonian(lh, ic_samp),
            'n_samples': n_samples,
            'mirror_samples': True
        }
        return ift.MetricGaussianKL(**dct)

    def get_calibration_kl(self, n_samples, cg_samp):
        lh = self.lhm_c.get_full()
        ic_samp = ift.GradientNormController(iteration_limit=cg_samp)
        dct = {
            'mean': self.position.extract(lh.domain),
            'hamiltonian': ift.StandardHamiltonian(lh, ic_samp),
            'n_samples': n_samples,
            'mirror_samples': True
        }
        return ift.MetricGaussianKL(**dct)

    @property
    def lhm_c(self):
        calib_sky = calibrator_sky(self.dh_t.sky_domain, 1.)
        return LikelihoodMaker(self.dh_c, calib_sky, self.cal_ops)

    def plot_overview(self, name, position=None):
        # FIXME Unify with Imager.plot_overview
        raise NotImplementedError
        if position is None:
            position = self.position
        p = ift.Plot()
        dct = {
            'aspect': 'auto',
            'xlabel': 'Time [s]',
            'ylabel': 'Antenna',
        }
        phfunc = lambda x: tuple_to_image(180/np.pi*x.force(position))
        amplfunc = lambda x: tuple_to_image(x.exp().force(position))
        pspecs = {}
        for key, op in self.cal_ops.items():
            pspecs[key] = op.pspec.force(position)
            if key[:-1] == 'ph':
                # p.add(phfunc(op.nozeropad), title=key, **dct)
                p.add(phfunc(op), title=key, **dct)
            if key[:-1] == 'ampl':
                p.add(amplfunc(op), title=key, **dct)
        p.add(pspecs.values(),
              label=list(pspecs.keys()),
              xlabel='Frequency [1/s]',
              title='Power spectra of calibration solutions')
        for key, lhm in self.likelihoods.items():
            lhm.plot_adjoint(
                p, 'Adjoint calibration distributor ({})'.format(key))
        for key, lhm in self.likelihoods.items():
            j = lhm.information_source(position)
            p.add(j, title='Calibrated Information Source {}'.format(key))

    def minimize_full_kl(self, n_samples, cg_samp):
        e = self.get_full_kl(n_samples, cg_samp)
        e, _ = self._minimizer(e)
        self.update_position(e.position)

    def minimize_calibration_kl(self, n_samples, cg_samp):
        e = self.get_calibration_kl(n_samples, cg_samp)
        e, _ = self._minimizer(e)
        self.update_position(e.position)

    def get_imaging_kl(self, n_samples, cg_samp, constant=''):
        lh = self.lhm_t.get_full()
        if self._isset_calibrator:
            calibration_domain = self.lhm_c.get_full().domain
            pos = self.position.extract(calibration_domain)
            _, lh = lh.simplify_for_constant_input(pos)
        ic_samp = ift.AbsDeltaEnergyController(
            0.5,
            convergence_level=5,
            iteration_limit=2000,
            file_name='{}energy_sampling.log'.format(self._prefix))
        dct = {
            'mean': self.position.extract(lh.domain),
            'hamiltonian': ift.StandardHamiltonian(lh, ic_samp),
            'n_samples': n_samples,
            'mirror_samples': True,
            'napprox': 20
        }
        if 'points' in self.position:
            dct['point_estimates'] = 'points'
        if constant == 'diffuse':
            dct['point_estimates'] = [
                'diffuse xi', 'diffuse phi', 'diffuse tau'
            ]
            dct['constants'] = ['diffuse xi', 'diffuse phi', 'diffuse tau']
            if 'points' in self.position:
                dct['point_estimates'].append('points')
        if constant == 'points':
            dct['point_estimates'] = ['points']
            dct['constants'] = ['points']
        if self._calibration_active:
            # FIXME Why is this necessary in the first place?
            cal_keys = []
            for op in self.cal_ops.values():
                cal_keys = cal_keys + list(op.domain.keys())
            cal_keys = list(set(cal_keys))
            dct['point_estimates'] = cal_keys
            dct['constants'] = cal_keys
        e = ift.MetricGaussianKL(**dct)
        from time import time
        self.plot_samples([e.position + samp for samp in e.samples],
                          'samples{}'.format(int(time())))
        return e

    # def _ig_imaging():
    #     vis, invvar = self.lhm_t.calibrated_visinvvar(self.position)
    #     if self.dh_t.active_wplanes > 1:
    #         raise RuntimeError
    #     lh = ift.GaussianEnergy(
    #         inverse_covariance=ift.makeOp(invvar)) @ ift.Adder(
    #             vis, neg=True) @ self.dh_t.R(0) @ sky

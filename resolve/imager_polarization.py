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

import configparser

import matplotlib.pyplot as plt
import numpy as np

import nifty5 as ift

from .amplitude_operators import SLAmplitude
from .constants import SPEEDOFLIGHT
from .diffuse import Diffuse, default_pspace
from .imager import Imager
from .sugar import save_pickle


class PolarizationImager(Imager):
    # FIXME Add possibility to set V to zero
    def __init__(self, output_path, iteration_limit, faraday_depth=0):
        super(PolarizationImager, self).__init__(output_path, iteration_limit)
        self._faraday_depth = float(faraday_depth)

    def _init_sky(self, config_parser, data_handler):
        sky_space = data_handler.sky_domain
        dom = sky_space[1]
        ad = {
            'n_pix': 64,
            'target': default_pspace(dom),
            'keys': ['diffuse tau', 'diffuse phi']
        }
        a = SLAmplitude(**dict(config_parser['diffuse power spectrum']), **ad)
        # FIXME Unify into one model
        dd = {'target': dom, 'amplitude': a}
        i = Diffuse(**dict(config_parser['diffuse']), **dd, xi_key='xi_i')
        ad['keys'] = ['linear pol diffuse tau', 'linear pol diffuse phi']
        dd['amplitude'] = SLAmplitude(
            **dict(
                config_parser['diffuse power spectrum linear polarization']),
            **ad)
        log_q = Diffuse(**dict(config_parser['diffuse linear polarization']),
                        **dd,
                        xi_key='xi_q').log_op
        log_u = Diffuse(**dict(config_parser['diffuse linear polarization']),
                        **dd,
                        xi_key='xi_u').log_op
        ad['keys'] = ['circular pol diffuse tau', 'circular pol diffuse phi']
        dd['amplitude'] = SLAmplitude(
            **dict(
                config_parser['diffuse power spectrum circular polarization']),
            **ad)
        log_v = Diffuse(**dict(config_parser['diffuse circular polarization']),
                        **dd,
                        xi_key='xi_v').log_op
        log_p = (log_q**2 + log_u**2 + log_v**2)**0.5

        self.I = i*log_p.cosh()
        self.Q = i*log_p.sinh()*log_p**-1*log_q
        self.U = i*log_p.sinh()*log_p**-1*log_u
        self.V = i*log_p.sinh()*log_p**-1*log_v

        # FIXME Make these separate fields
        I = ift.DomainTupleFieldInserter(sky_space, 0, (0,)) @ self.I
        Q = ift.DomainTupleFieldInserter(sky_space, 0, (1,)) @ self.Q
        U = ift.DomainTupleFieldInserter(sky_space, 0, (2,)) @ self.U
        V = ift.DomainTupleFieldInserter(sky_space, 0, (3,)) @ self.V

        self._diffuse = I + Q + U + V

        self.dh_t = data_handler
        self._update_position_domain()

    def initialize_target_skymodel(self, a, k0, sm, sv, im, iv, mean, stddev,
                                   data_handler):
        config = configparser.ConfigParser()
        config['diffuse power spectrum'] = {
            'a': a,
            'k0': k0,
            'sm': sm,
            'sv': sv,
            'im': im,
            'iv': iv,
        }
        config['diffuse power spectrum linear polarization'] = config[
            'diffuse power spectrum']
        config['diffuse power spectrum circular polarization'] = config[
            'diffuse power spectrum']
        config['diffuse'] = {'mean': mean, 'stddev': stddev}
        config['diffuse linear polarization'] = {
            'mean': 0,
            'stddev': 0.05*stddev
        }
        config['diffuse circular polarization'] = config[
            'diffuse linear polarization']
        self._init_sky(config, data_handler)

    def plot_samples(self, samples, name):
        p = ift.Plot()
        for samp in samples:
            p.add(self.I.force(samp), title='Stokes I')
            p.add(self.Q.force(samp), title='Stokes Q')
            p.add(self.U.force(samp), title='Stokes U')
            p.add(self.V.force(samp), title='Stokes V')
        p.output(name='{}{}.png'.format(self._prefix, name),
                 xsize=40,
                 ysize=20,
                 nx=8)

    def save(self, name, position=None):
        pos = self.position if position is None else position
        save_pickle(
            pos, '{}{:02}_{}.pickle'.format(self._prefix, self._counter, name))
        fname = '{:02}_{}'.format(self._counter, name)
        p = ift.Plot()
        p.add(self.I.force(pos), title='Stokes I')

        fld = self.Q.force(pos)
        lim = np.max(np.abs(fld.to_global_data()))
        p.add(fld, title='Stokes Q', zmin=-lim, zmax=lim)

        fld = self.U.force(pos)
        lim = np.max(np.abs(fld.to_global_data()))
        p.add(fld, title='Stokes U', zmin=-lim, zmax=lim)

        fld = self.V.force(pos)
        lim = np.max(np.abs(fld.to_global_data()))
        p.add(fld, title='Stokes V', zmin=-lim, zmax=lim)

        p.add((self.Q**2 + self.U**2 + self.V**2).sqrt().force(pos),
              title='Polarized intensity')
        p.add((abs(self.U.force(pos))**2 + abs(self.Q.force(pos))**2 +
               abs(self.V.force(pos))**2).sqrt()/self.I.force(pos),
              title='Fractional Polarization')
        p.output(name='{}{}.png'.format(self._prefix, fname),
                 nx=2,
                 xsize=20,
                 ysize=30)

        # FIXME Move into data handler and support polarization
        self.dh_t.writefits(
            self.I.force(pos),
            '{}{:02}_{}_I.fits'.format(self._prefix, self._counter, name))
        self.dh_t.writefits(
            self.Q.force(pos),
            '{}{:02}_{}_Q.fits'.format(self._prefix, self._counter, name))
        self.dh_t.writefits(
            self.U.force(pos),
            '{}{:02}_{}_U.fits'.format(self._prefix, self._counter, name))
        self.dh_t.writefits(
            self.V.force(pos),
            '{}{:02}_{}_V.fits'.format(self._prefix, self._counter, name))

        np.seterr(under='ignore')
        npix = self.diffuse.target.shape[1]
        x = np.arange(0, npix)
        y = np.arange(0, npix)
        X, Y = np.meshgrid(x, y)
        f = np.rot90
        phi = f(
            np.arctan((self.U.force(pos)/self.Q.force(pos)).to_global_data()))
        length = f((self.Q**2 + self.U**2).sqrt().force(pos).to_global_data())

        freq = np.mean(self.dh_t.freqs)
        faraday_correction = self._faraday_depth*(1/freq*SPEEDOFLIGHT)**2

        Ex = length*np.sin(phi + np.pi/2 + faraday_correction)
        Ey = length*np.cos(phi + np.pi/2 + faraday_correction)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.streamplot(x,
                      y,
                      Ex,
                      Ey,
                      density=6,
                      color=length,
                      cmap=plt.cm.Blues,
                      arrowstyle='-',
                      linewidth=0.5)
        ax.contour(x,
                   y,
                   f(self.I.log().force(pos).to_global_data()),
                   linewidths=.5,
                   cmap=plt.cm.Reds)
        ax.set_aspect('equal')
        plt.title('Faraday rotation correction: RM = {} rad/m²'.format(
            self._faraday_depth))
        plt.savefig('{}{}_pol.png'.format(self._prefix, fname))
        plt.close()
        np.seterr(under='raise')

        self._counter += 1

    def find_parameters(self,
                        sm,
                        plotting=False,
                        start=2,
                        stop=30,
                        step=1,
                        alpha=0.5):
        raise NotImplementedError

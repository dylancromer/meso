import pytest
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1, rc={"lines.linewidth": 2, 'lines.markersize': 8.0,})
import scipy.interpolate
import meso


COLORS = ['#5599e2', '#de6957', '#73bc77']
NUM_MC = 40000
SCALE = 0.1
SEED = 133
PLOT_DIR = 'figs/test/'


def get_arg(r, off, phi):
    r = r[:, None]
    off = off[None, :]
    phi = phi[None, :]
    return np.sqrt(r**2 + off**2 - 2*r*off*np.cos(phi))


def _rayleigh_dist(x, scale):
    scale = meso.mathutils.atleast_kd(scale, x.ndim+1, append_dims=False)
    x = x[..., None]
    return (x/scale**2) * np.exp(-(x/scale)**2 / 2)


def rayleigh_dist(x):
    return _rayleigh_dist(x, scale=np.array([SCALE]))


def describe_miscentering():

    @pytest.fixture
    def radial_grid():
        return np.geomspace(0.0003, 300, 600)

    @pytest.fixture
    def density():
        return np.loadtxt('int/data/rho_gnfw_for_meso_test.txt.gz')/1e14

    @pytest.fixture
    def rng():
        return np.random.default_rng(seed=SEED)

    def it_matches_a_monte_carlo_miscentered_profile(radial_grid, density, rng):
       rho_func = scipy.interpolate.interp1d(radial_grid, density, kind='cubic')

       rs = np.geomspace(0.01, 100, 100)
       offsets = rng.rayleigh(scale=SCALE, size=NUM_MC)
       phis = 2*np.pi*rng.random(size=NUM_MC)
       rho_arg = get_arg(rs, offsets, phis)
       idx = np.argwhere(rho_arg < radial_grid.min())
       rho_arg = np.delete(rho_arg, idx[:, 1:], axis=1)

       plt.hist(
           np.log(np.repeat(rs, NUM_MC)),
           bins=48,
           histtype='stepfilled',
           color='black',
           alpha=0.8,
           label='radius',
       )
       plt.hist(
           np.log(rho_arg).flatten(),
           bins=48,
           color=COLORS[0],
           histtype='stepfilled',
           alpha=0.8,
           label='miscentered radius',
       )
       plt.xlabel(r'$\ln(r)$')
       plt.ylabel('number')
       plt.legend(loc='best')
       plt.gcf().set_size_inches(6, 4)
       plt.savefig(PLOT_DIR + 'radius_hist.svg', bbox_inches='tight')
       plt.gcf().clear()

       rhos = rho_func(rs)
       rhos_mc = rho_func(rho_arg).mean(axis=(1))
       meso_rhos_coarse = meso.Rho(
            num_offset_radii=50,
            num_phis=4,
        ).miscenter(rs, rho_func, prob_dist_func=rayleigh_dist)
       meso_rhos_medium = meso.Rho(
            num_offset_radii=100,
            num_phis=12,
        ).miscenter(rs, rho_func, prob_dist_func=rayleigh_dist)
       meso_rhos_fine = meso.Rho(
            num_offset_radii=200,
            num_phis=36,
        ).miscenter(rs, rho_func, prob_dist_func=rayleigh_dist)

       plt.plot(rs, rhos, color='black', label='centered')
       plt.plot(rs, meso_rhos_coarse, color=COLORS[1], alpha=0.8, linestyle=':', label='coarse')
       plt.plot(rs, meso_rhos_medium, color=COLORS[1], alpha=0.8, linestyle='--', label='medium')
       plt.plot(rs, meso_rhos_fine, color=COLORS[1], alpha=0.8, linestyle='-', label='fine')
       plt.plot(rs, rhos_mc, color=COLORS[0], alpha=0.8, label='Monte Carlo')
       plt.xscale('log')
       plt.yscale('log')
       plt.xlabel(r'$r$ (Mpc)')
       plt.ylabel(r'$\rho(r)$')
       plt.legend(loc='best')
       plt.gcf().set_size_inches(6, 4)
       plt.savefig(PLOT_DIR + 'rho_comparison.svg', bbox_inches='tight')
       plt.gcf().clear()

       plt.plot(rs, rs**2 * rhos, color='black', label='centered')
       plt.plot(rs, rs**2 * meso_rhos_coarse, color=COLORS[1], alpha=0.8, linestyle=':', label='coarse')
       plt.plot(rs, rs**2 * meso_rhos_medium, color=COLORS[1], alpha=0.8, linestyle='--', label='medium')
       plt.plot(rs, rs**2 * meso_rhos_fine, color=COLORS[1], alpha=0.8, linestyle='-', label='fine')
       plt.plot(rs, rs**2 * rhos_mc, color=COLORS[0], alpha=0.8, label='Monte Carlo')
       plt.xscale('log')
       plt.xlabel(r'$r$ (Mpc)')
       plt.ylabel(r'$r^2 \rho(r)$')
       plt.legend(loc='best')
       plt.gcf().set_size_inches(6, 4)
       plt.savefig(PLOT_DIR + 'rho_comparison_scaled.svg', bbox_inches='tight')
       plt.gcf().clear()

from dataclasses import dataclass
import numpy as np
import meso.mathutils as mathutils


@dataclass
class Rho:
    num_offset_radii: int = 160
    num_phis: int = 12
    min_radius: float = 1e-4
    max_radius: float = 10

    def _rho_arg(self, radii, ys, phis):
        ys_ = mathutils.atleast_kd(ys, radii.ndim+2)
        phis_ = mathutils.atleast_kd(phis[None, :], radii.ndim+2)
        radii_ = radii[None, None, ...]
        return np.sqrt(radii_**2 + ys_**2 + 2*radii_*ys_*np.cos(phis_))

    def _get_y_samples(self):
        return np.logspace(
            np.log10(self.min_radius),
            np.log10(self.max_radius),
            self.num_offset_radii,
        )

    def _get_phi_samples(self):
        return np.linspace(0, 2*np.pi, self.num_phis)

    def _reshape_probs(self, probs, radii_ndim, rho_ndim):
        return np.moveaxis(
            mathutils.atleast_kd(probs, rho_ndim),
            1,
            -1,
        )

    def _get_y_phi_integrand_and_differentials(self, radii, rho_func, prob_dist_func):
        ys = self._get_y_samples()
        phis = self._get_phi_samples()

        rho_arg = self._rho_arg(radii, ys, phis)
        rhos = rho_func(rho_arg)
        probs = self._reshape_probs(prob_dist_func(ys), radii.ndim, rhos.ndim)
        return probs * rhos, np.gradient(ys), np.gradient(phis)

    def miscenter(self, radii, rho_func, prob_dist_func):
        y_phi_integrand, dys, dphis = self._get_y_phi_integrand_and_differentials(radii, rho_func, prob_dist_func)
        dys_ = mathutils.atleast_kd(dys, y_phi_integrand.ndim)
        dphis_ = mathutils.atleast_kd(dphis, y_phi_integrand.ndim)
        phi_integrand = mathutils.trapz_(y_phi_integrand, axis=0, dx=dys)
        return mathutils.trapz_(phi_integrand, axis=0, dx=dphis) / (2*np.pi)

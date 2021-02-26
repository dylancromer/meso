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
        ys = mathutils.atleast_kd(ys[:, None], radii.ndim+2, append_dims=False)
        phis = mathutils.atleast_kd(phis, radii.ndim+2, append_dims=False)
        radii = radii[..., None, None]
        return np.sqrt(radii**2 + ys**2 + 2*radii*ys*np.cos(phis))

    def _get_y_samples(self):
        return np.logspace(
            np.log10(self.min_radius),
            np.log10(self.max_radius),
            self.num_offset_radii,
        )

    def _get_phi_samples(self):
        return np.linspace(0, 2*np.pi, self.num_phis)

    def _reshape_probs(self, probs, radii_ndim, rho_ndim):
        probs = mathutils.atleast_kd(
            probs,
            probs.ndim+radii_ndim,
            append_dims=False,
        )
        return mathutils.atleast_kd(
            probs,
            rho_ndim,
        )

    def _get_y_phi_integrand_and_differentials(self, radii, rho_func, prob_dist_func):
        ys = self._get_y_samples()
        phis = self._get_phi_samples()

        rho_arg = self._rho_arg(radii, ys, phis)
        rhos = rho_func(rho_arg)
        probs = self._reshape_probs(prob_dist_func(ys), radii.ndim, rhos.ndim)
        probs = np.moveaxis(
            probs,
            ys.ndim+radii.ndim,
            -1,
        )
        return probs * rhos, np.gradient(ys), np.gradient(phis)

    def miscenter(self, radii, rho_func, prob_dist_func):
        y_phi_integrand, dys, dphis = self._get_y_phi_integrand_and_differentials(radii, rho_func, prob_dist_func)
        phi_integrand = mathutils.trapz_(y_phi_integrand, radii.ndim, dx=dys)
        return mathutils.trapz_(phi_integrand, radii.ndim, dx=dphis) / (2*np.pi)

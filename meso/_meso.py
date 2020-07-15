from dataclasses import dataclass
import numpy as np
import meso.mathutils as mathutils


@dataclass
class Rho:
    NUM_PHIS = 40

    num_points: int = 200
    min_radius: float = 1e-4
    max_radius: float = 10

    def _rho_arg(self, radii, ys, phis):
        radii = radii[:, None, None]
        ys = ys[None, :, None]
        phis = phis[None, None, :]
        return np.sqrt(radii**2 + ys**2 + 2*radii*ys*np.cos(phis))

    def _get_y_samples(self):
        return np.logspace(
            np.log10(self.min_radius),
            np.log10(self.max_radius),
            self.num_points,
        )

    def _get_phi_samples(self):
        return np.linspace(0, 2*np.pi, self.NUM_PHIS)

    def _get_y_phi_integrand_and_differentials(self, radii, rho_func, prob_dist_func):
        ys = self._get_y_samples()
        phis = self._get_phi_samples()

        rho_arg = self._rho_arg(radii, ys, phis)
        rhos = rho_func(rho_arg)
        probs = mathutils.atleast_kd(
            prob_dist_func(ys)[None, :, None],
            rhos.ndim,
        )
        return probs * rhos, np.gradient(ys), np.gradient(phis)

    def miscenter(self, radii, rho_func, prob_dist_func):
        y_phi_integrand, dys, dphis = self._get_y_phi_integrand_and_differentials(radii, rho_func, prob_dist_func)
        phi_integrand = mathutils.trapz_(y_phi_integrand, 1, dx=dys)
        return mathutils.trapz_(phi_integrand, 1, dx=dphis) / (2*np.pi)

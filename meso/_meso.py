from dataclasses import dataclass
import numpy as np
import meso.mathutils as mathutils


NUM_PHIS = 40


@dataclass
class Rho:
    min_radius: float = 1e-4
    max_radius: float = 10

    def _rho_arg(self, radii, ys, phis):
        radii = radii[:, None, None]
        ys = ys[None, :, None]
        phis = phis[None, None, :]
        return np.sqrt(radii**2 + ys**2 + 2*radii*ys*np.cos(phis))

    def miscenter(self, radii, rho_func, prob_dist_func, num_points=200):
        ys = np.logspace(
            np.log10(self.min_radius),
            np.log10(self.max_radius),
            num_points,
        )

        phis = np.linspace(0, 2*np.pi, NUM_PHIS)

        rho_arg = self._rho_arg(radii, ys, phis)
        rhos = rho_func(rho_arg)
        probs = mathutils.atleast_kd(
            prob_dist_func(ys)[None, :, None],
            rhos.ndim,
        )
        y_integrand = probs * rhos

        phi_integrand = mathutils.trapz_(y_integrand, 1, dx=np.gradient(ys))
        return mathutils.trapz_(phi_integrand, 1, dx=np.gradient(phis)) / (2*np.pi)

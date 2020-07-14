import pytest
import numpy as np
import meso
import meso.mathutils as mathutils


def describe_miscenter():

    @pytest.fixture
    def miscenter_model():
        return meso.Rho()

    def it_calculates_the_aggregate_miscentered_density(miscenter_model):
        rs = np.logspace(-2, 2, 30)
        def rho_func(x): return 1/x**2

        sigma = 0.5
        def prob_dist_func(r): return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(r-0.2)**2/(2*sigma**2))

        rho_miscs = miscenter_model.miscenter(rs, rho_func, prob_dist_func)

        assert not np.any(np.isnan(rho_miscs))

    def it_can_handle_larger_shapes(miscenter_model):
        rs = np.logspace(-2, 2, 30)
        slopes = np.linspace(2, 3, 8)
        amps = np.linspace(0.9, 1.1, 5)
        def rho_func(x):
            amps_ = mathutils.atleast_kd(amps, x.ndim+2, append_dims=False)
            slopes_ = mathutils.atleast_kd(slopes[:, None], x.ndim+2, append_dims=False)
            return amps_/x[..., None, None]**slopes_

        sigma = 0.5
        def prob_dist_func(r): return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(r-0.2)**2/(2*sigma**2))

        rho_miscs = miscenter_model.miscenter(rs, rho_func, prob_dist_func)

        assert not np.any(np.isnan(rho_miscs))
        assert rho_miscs.shape == (30, 8, 5)
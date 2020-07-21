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
        def rho_func(x):
            return np.ones(x.shape+(1,))/x[..., None]**2

        def prob_dist_func(r):
            sigmas = np.array([0.4, 0.5])
            sigmas = mathutils.atleast_kd(sigmas, r.ndim+1, append_dims=False)
            r = mathutils.atleast_kd(r, r.ndim+1)
            return 1/np.sqrt(2*np.pi*sigmas**2) * np.exp(-(r-0.2)**2/(2*sigmas**2))

        rho_miscs = miscenter_model.miscenter(rs, rho_func, prob_dist_func)

        assert not np.any(np.isnan(rho_miscs))

    def it_can_handle_larger_shapes(miscenter_model):
        rs = np.logspace(-2, 2, 30)
        slopes = np.linspace(2, 3, 8)
        amps = np.linspace(0.9, 1.1, 2)
        def rho_func(x):
            amps_ = mathutils.atleast_kd(amps, x.ndim+2, append_dims=False)
            slopes_ = mathutils.atleast_kd(slopes[:, None], x.ndim+2, append_dims=False)
            return amps_/x[..., None, None]**slopes_

        def prob_dist_func(r):
            sigmas = np.array([0.4, 0.5])
            sigmas = mathutils.atleast_kd(sigmas, r.ndim+1, append_dims=False)
            r = mathutils.atleast_kd(r, r.ndim+1)
            return 1/np.sqrt(2*np.pi*sigmas**2) * np.exp(-(r-0.2)**2/(2*sigmas**2))

        rho_miscs = miscenter_model.miscenter(rs, rho_func, prob_dist_func)

        assert not np.any(np.isnan(rho_miscs))
        assert rho_miscs.shape == (30, 8, 2)

    def it_gives_the_right_answer_for_an_analytical_example(miscenter_model):
        rs = np.logspace(-2, 2, 30)
        def rho_func(x): return x**2

        ul = miscenter_model.max_radius
        ll = miscenter_model.min_radius
        interval = ul - ll
        def prob_dist_func(r): return np.ones(r.shape+(1,))/interval

        rho_miscs = miscenter_model.miscenter(rs, rho_func, prob_dist_func).squeeze()

        analytical_answer = 1/3 * (ll**2 + ul**2 + ul*ll + 3*rs**2)

        assert np.allclose(rho_miscs, analytical_answer, rtol=1e-3)

    def it_can_handle_larger_radii_shapes(miscenter_model):
        rs = np.logspace(-2, 2, 30)
        rs = np.stack((rs, rs, rs))
        slopes = np.linspace(2, 3, 8)
        amps = np.linspace(0.9, 1.1, 2)
        def rho_func(x):
            amps_ = mathutils.atleast_kd(amps, x.ndim+2, append_dims=False)
            slopes_ = mathutils.atleast_kd(slopes[:, None], x.ndim+2, append_dims=False)
            return amps_/x[..., None, None]**slopes_

        def prob_dist_func(r):
            sigmas = np.array([0.4, 0.5])
            sigmas = mathutils.atleast_kd(sigmas, r.ndim+1, append_dims=False)
            r = mathutils.atleast_kd(r, r.ndim+1)
            return 1/np.sqrt(2*np.pi*sigmas**2) * np.exp(-(r-0.2)**2/(2*sigmas**2))

        rho_miscs = miscenter_model.miscenter(rs, rho_func, prob_dist_func)

        assert not np.any(np.isnan(rho_miscs))
        assert rho_miscs.shape == (3, 30, 8, 2)

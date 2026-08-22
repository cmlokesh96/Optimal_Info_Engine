#This file defines a class, reminiscent of a C struct containing all the relevant variables determining the system
#The explicit typing and creation using a create instead of the init method makes it compatible with njit
from typing import NamedTuple
from scipy import constants
from numpy import sqrt
class SystemVariables(NamedTuple):
    kappa: float
    tau_b: float
    tau_p: float
    kappa_b: float
    T: float
    x_thresh: float
    gamma: float
    gamma_b: float
    k: float
    k_B: float

    @classmethod
    def create(cls, gamma, gamma_b, kappa, kappa_b, T, x_thresh_sigma_multiple =1.54, k_B=1):
        tau_p = gamma/kappa_b
        tau_b = gamma_b/kappa_b
        x_thresh = sqrt(k_B * T/kappa) * x_thresh_sigma_multiple
        k = kappa / kappa_b
        return cls(kappa, tau_b, tau_p, kappa_b, T, x_thresh, gamma, gamma_b, k, k_B)

#params_bath_model = SystemVariables.create(tau_b=1.1, tau_p=1, kappa=1, kappa_b=0.75, T=1, x_thresh=0.5)
params_bath_model = SystemVariables.create(kappa=2e-6, kappa_b=1e-6, gamma=0.34e-6, gamma_b =15e-6, T=298, k_B= constants.Boltzmann)


def length_scale(params):
    """
    Equilibrium length scale sqrt(k_B*T/kappa) for `params`.

    This is the standard deviation of the equilibrium position distribution
    (equipartition theorem), and the conversion factor between a
    nondimensional position (in units where this scale is 1) and the
    corresponding position in whatever units `params` uses (e.g. SI meters).

    Parameters
    ----------
    params : SystemVariables

    Returns
    -------
    float
        sqrt(k_B * T / kappa), in the length units implied by `params`.
    """
    return sqrt(params.k_B * params.T / params.kappa)


def energy_scale(params):
    """
    Thermal energy scale k_B*T for `params`.

    Conversion factor between a nondimensional energy/work value (in units
    of k_B*T) and the corresponding value in whatever units `params` uses.

    Parameters
    ----------
    params : SystemVariables

    Returns
    -------
    float
        k_B * T, in the energy units implied by `params`.
    """
    return params.k_B * params.T


def make_nondimensional(params):
    """
    Derive a nondimensional counterpart of `params`, with k_B = T = kappa = 1
    (i.e. lengths measured in units of sqrt(k_B*T/kappa), so the equilibrium
    standard deviation of position is exactly 1).

    The dimensionless ratio k = kappa/kappa_b and the physical relaxation
    timescales tau_b, tau_p are preserved exactly (time is left in the
    original units - e.g. seconds - since fixing the length scale to O(1)
    does not require rescaling time). gamma, gamma_b, kappa_b are re-derived
    from k, tau_b, tau_p so that these ratios come out unchanged. The
    relative threshold position x_thresh/sigma_eq is also preserved, so
    outcome-region boundaries (via `discretization_edges`/`set_bounds`) mean
    the same thing physically in both unit systems.

    Parameters
    ----------
    params : SystemVariables
        Original parameter set (typically SI-scale).

    Returns
    -------
    SystemVariables
        Nondimensional counterpart with k_B = T = kappa = 1, same k,
        tau_b, tau_p, and same relative threshold as `params`.
    """
    kappa_nondim = 1.0
    kappa_b_nondim = kappa_nondim / params.k
    gamma_nondim = params.tau_p * kappa_b_nondim
    gamma_b_nondim = params.tau_b * kappa_b_nondim

    x_thresh_sigma_multiple = params.x_thresh / length_scale(params)

    return SystemVariables.create(
        gamma=gamma_nondim,
        gamma_b=gamma_b_nondim,
        kappa=kappa_nondim,
        kappa_b=kappa_b_nondim,
        T=1.0,
        k_B=1.0,
        x_thresh_sigma_multiple=x_thresh_sigma_multiple,
    )


# Nondimensional counterpart of params_bath_model, recomputed automatically
# from it - if params_bath_model above changes, this changes with it 
params_bath_model_nondim = make_nondimensional(params_bath_model)
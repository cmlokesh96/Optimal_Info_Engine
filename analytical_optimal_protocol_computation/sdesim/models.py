"""precompiled right hand sides determining the deterministig change of the positional variables""" 
import numpy as np
from numba import njit

@njit(cache=True)
def rhs_single_particle_in_well(state, current_lambda, parameters):
    """right hand side for OUP.

    Parameters:
    state: array like state vector containing X
    current_lambda: position of trap at evaluation
    parameters: SytemVariables object containing variables determining system

    Returns:
    array like containing the derivatives of the entrie(s) of state according to ODE"""
    return np.array([-parameters.kappa/parameters.gamma *(state[0]-current_lambda)])

@njit(cache=True)
def rhs_bath_model(state,current_lambda,parameters):
    """right hand side for the bath model.

    Parameters:
    state: array like state vector containing X, X_b
    current_lambda: position of trap at evaluation
    parameters: SytemVariables object containing variables determining system

    Returns:
    array like containing the derivatives of the entrie(s) of state according to ODE"""
    X_change = -parameters.kappa/parameters.gamma *(state[0]-current_lambda) - 1/parameters.tau_p * (state[0]- state[1])
    Xb_change = 1/parameters.tau_b * (state[0]- state[1])
    return np.array([X_change,Xb_change])
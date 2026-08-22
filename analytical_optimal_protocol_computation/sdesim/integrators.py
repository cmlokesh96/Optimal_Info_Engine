#File that does the actual (optimized) simulation of the trajectories 
import numpy as np
from numba import njit, prange
from .models import rhs_single_particle_in_well, rhs_bath_model
from .analytics import make_optimal_protocol_numba
from .helpers import reshape_constant_initial_state

#
@njit(cache=True) #This flags the following function for precompilation
def default_protocol(t, parameters):
    """The simulations should be possible during execution of an protocol, so it expects one as an argument. As we also want to simulate the case before start of the protocol,
        the default is just constant 0, put in a seperate function, so it can be precompiled by njit as well

        Parameters:
        parameters: Systemvariables
        t: float, time to evaluate protocol

        returns: float, trap focus at time t"""
    return 0.0

#As njit cannot handle numpy random generators, but only the deprecated np.random(), there is a wrapper that generates the necessary random numbers (in one call
#for all trajectories as one vectorized call), which then calls the actual integration of all trajectories in batch, so parallezation with numba can actualy be fruitful
#The structure is implement twice, once for each langevin equation as numba apparently sometimes behaves unexpectedly for variable function with unknown parameter types
@njit(parallel=True, cache=True)
def integrate_batch_single_particle_in_well(noise_amplitude, initial_states, time_step, random_numbers, parameters, protocol):
    """Advances every sample by steps_per_save fine steps, slightly outdated, as it's storage of intermediate steps does not scale well with limited memory,
    yet only for testing, so not of great relevance
    
    Parameters:
    noise_amplitude: np.array containing the prefactor of the random numbers for each entry of a current_state, i.e X,X_b
    initial_states: np.array containing arrays of initial conditions for each trajectory, so they can be created in one call
    t_final: float, time to simulate up to
    time_step: float, timestep of simulation 
    random_numbers: shape (number_of_samples, steps_per_save, dimension), already scaled by sqrt(dt).
    parameters: SystemVariables 
    protocol: function with arguments t, parameters that gives trap position at thaat time, for out of equilibrium considerations

    Return array of all simulated trajectories at all simulated times

     """
    number_of_samples, number_of_steps, dimension = random_numbers.shape
    trajectories = np.empty((number_of_samples, dimension + 1, number_of_steps + 1))

    for l in prange(number_of_samples):#prange is for parallezation of for loops which here is possible as the trajectories do not depend on eachother
        current_state = initial_states[l].copy()
        t = 0.0
        trajectories[l, 0, 0] = t
        trajectories[l, 1:, 0] = current_state

        for i in range(number_of_steps): #looping over the timesteps
            t += time_step
            current_lambda = protocol(t, parameters)
            current_state = current_state + rhs_single_particle_in_well(current_state, current_lambda, parameters) * time_step + noise_amplitude * random_numbers[l, i]
            trajectories[l, 0, i + 1] = t
            trajectories[l, 1:, i + 1] = current_state

    return trajectories

#Wrappers for the random number generation. the noise amplitude should be an array of same shape as current shape and contain the prefactors of the random numbers used in the euler maruyama step
def euler_maruyama_ensemble_single_particle_in_well(dimension, noise_amplitude, initial_states, t_final, time_step, number_of_samples, random_number_generator, parameters, protocol=default_protocol):
    """Simulates number_of_samples trajectories til final time, wraps the integrate_batch_single_particle_in_well to generate random numbers outside of numba
    
    Parameters:
    
    dimension: int, number of variables describing state
    noise_amplitude: np.array containing the prefactor of the random numbers for each entry of a current_state, i.e X,X_b
    initial_states: np.array containing arrays of initial conditions for each trajectory, so they can be created in one call
    t_final: float, time to simulate up to
    time_step: float, time_step for actual simulation
    number_of_samples: int, how many trajectories to simulate
    random_number_generator: numpy.random rng, preinitialized for compareability
    parameters: Systemvariables
    protocol: function with arguments t, parameters that gives trap position at thaat time, for out of equilibrium considerations

    Return: array of all simulated trajectories at all simulated times"""
    number_of_steps = int(np.ceil(t_final/time_step))
    sqrt_dt = np.sqrt(time_step) #Standard derivation of the random numbers in the euler step

    random_numbers = random_number_generator.normal(0.0, sqrt_dt, size=(number_of_samples, number_of_steps, dimension))

    return integrate_batch_single_particle_in_well(noise_amplitude, initial_states, time_step, random_numbers, parameters, protocol)



@njit(parallel=True, cache=True)
def advance_batch_bath_model(current_states, noise_amplitude, time_step, random_numbers, parameters, protocol, t_start):
    """
    Advances every sample by steps_per_save fine steps, without storing intermediate states.

    Parameters:
    current_states: np array containing number_of_samples np.arrays of length dimension, containing current X,X_b for each simultaneosly simulated trajectory
    noise_amplitude: np.array containing the prefactor of the random numbers for each entry of a current_state, i.e X,X_b
    time_step: float, time-step actually used in simulation, not in saving
    random_numbers: shape (number_of_samples, steps_per_save, dimension), already scaled by sqrt(dt).
    parameters: SystemVariables 
    protocol: function with arguments t, parameters that gives trap position at thaat time, for out of equilibrium considerations
    t_start: float, initial time at beginning of this batch
    Returns: the new states, shape (number_of_samples, dimension).
    """
    number_of_samples, steps_per_save, dimension = random_numbers.shape
    new_states = np.empty_like(current_states)

    for l in prange(number_of_samples):
        state = current_states[l].copy()
        t = t_start
        for i in range(steps_per_save):
            t += time_step
            current_lambda = protocol(t, parameters)
            state = state + rhs_bath_model(state, current_lambda, parameters) * time_step + noise_amplitude * random_numbers[l, i]
        new_states[l] = state

    return new_states


def euler_maruyama_ensemble_bath_model(dimension, noise_amplitude, initial_states, t_final,
                                        output_time_step, fine_time_step, number_of_samples,
                                        random_number_generator, parameters, protocol=default_protocol):
    """Simulates number_of_samples trajectories till final time, wraps the advance_batch_bath_model that does the actual stepping in a trade off between paralleliazability and memory
    
    Parameters:
    
    dimension: int, number of variables describing state
    noise_amplitude: np.array containing the prefactor of the random numbers for each entry of a current_state, i.e X,X_b
    initial_states: np.array containing arrays of initial conditions for each trajectory, so they can be created in one call
    t_final: float, time to simulate up to
    output_time_step: float, time after which current state is saved
    fine_time_step: time_step for actual simulation
    number_of_samples: int, how many trajectories to simulate
    random_number_generator: numpy.random rng, preinitialized for compareability
    parameters: Systemvariables
    protocol: function with arguments t, parameters that gives trap position at thaat time, for out of equilibrium considerations

    Returns:
    ndarray containing the trajectories in resolution output_time_step
    """
    steps_per_save = int(round(output_time_step / fine_time_step))
    if not np.isclose(steps_per_save * fine_time_step, output_time_step):
        raise ValueError("output_time_step must be an integer multiple of fine_time_step")

    number_of_saves = int(np.ceil(t_final / output_time_step))
    sqrt_dt = np.sqrt(fine_time_step)

    trajectories = np.empty((number_of_samples, dimension + 1, number_of_saves + 1))
    current_states = initial_states.copy()
    t = 0.0
    trajectories[:, 0, 0] = t
    trajectories[:, 1:, 0] = current_states

    # allocate exactly once, outside the loop
    random_numbers = np.empty((number_of_samples, steps_per_save, dimension))

    for k in range(number_of_saves):
        #compute random numbers outside of simulation as njit can't deal with modern numpy rngs
        random_number_generator.standard_normal(out=random_numbers)  # refills buffer in place
        random_numbers *= sqrt_dt                                     # scale in place, no new array

        current_states = advance_batch_bath_model(
            current_states, noise_amplitude, fine_time_step, random_numbers, parameters, protocol, t
        )
        t += output_time_step
        trajectories[:, 0, k + 1] = t
        trajectories[:, 1:, k + 1] = current_states

    return trajectories


def simulate_stochastic_trajectories_under_optimal_protocol(c_vector, lambda_f, t_protocol_end, params, number_of_samples, fine_time_step, random_number_generator, target_number_of_saves=1000):
    """
    Simulate an ensemble of stochastic trajectories under the
    optimal protocol lambda(t) all
    starting at the same deterministic initial condition
    (c_vector[0], c_vector[2]) 

    Parameters:
    c_vector : np.ndarray, shape (4,)
        As returned by get_cvector/get_cvector_numeric; only c_vector[0].
    lambda_f : float
    t_protocol_end : float
    params : SystemVariables
    number_of_samples : int
    fine_time_step : float
        Integration step 
    random_number_generator : numpy.random.Generator
    target_number_of_saves : int, optional
        Approximate number of stored points per trajectory, independent of
        t_protocol_end (output_time_step is derived from it). Default 1000.

    Returns
    -------
    np.ndarray
        Shape (number_of_samples, 3, number_of_saves+1); rows (t, X, X_b).
    """
    optimal_protocol = make_optimal_protocol_numba(c_vector, t_protocol_end, lambda_f)

    steps_per_save = max(1, round((t_protocol_end / fine_time_step) / target_number_of_saves))
    output_time_step = steps_per_save * fine_time_step

    noise_amplitude = np.sqrt(np.array([
        2 * params.T * params.k_B / params.gamma,
        2 * params.T * params.k_B / params.gamma_b,
    ]))

    initial_conditions = reshape_constant_initial_state(
        np.array([c_vector[0], c_vector[2]]), number_of_samples
    )

    return euler_maruyama_ensemble_bath_model(
        2, noise_amplitude, initial_conditions, t_protocol_end, output_time_step,
        fine_time_step, number_of_samples, random_number_generator, params,
        protocol=optimal_protocol,
    )
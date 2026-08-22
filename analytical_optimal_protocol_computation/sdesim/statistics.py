#This file is dedicated to explicitly computing statistical quantities/metrics
import numpy as np
from scipy import stats
from scipy.integrate import dblquad, tplquad
from sdesim.helpers import time_to_step_map, discretization_edges, assign_outcome_labels, outcome_labels, set_bounds, finite_bound, format_trajectories_at_two_times


def OUP_stationary_autocorrrelation(time_difference, parameters):
    """Compute the analytical autocorrelation at given time.

    Parameters:
    parameters: SystemVariables
    time_difference: float
    return: float, <X(0)X(t)>"""
    return (parameters.k_B*parameters.T)/(parameters.kappa)*np.exp(-parameters.kappa/parameters.gamma*time_difference)

#trajectories_formatted should be an array containing 1 array for each variable as a row
def fit_gaussian(trajectories_formatted, random_number_generator):
    #mean and cov fully determine a gaussian, so the computation of them is the fitting
    means = np.mean(trajectories_formatted, axis=1)
    covariance = np.cov(trajectories_formatted)

    return stats.multivariate_normal(means, covariance,seed=random_number_generator)


def correlation_to_init_time(trajectories, t, timestep, variable_indices=np.array([1,1])):
    index = time_to_step_map(t,timestep)
    positions_at_0 = trajectories[:, variable_indices[0], 0]
    positions_at_t = trajectories[:, variable_indices[1], index]
    return np.mean(positions_at_0 * positions_at_t)

def stationary_correlation(trajectories, time_difference, time_step, t_transient):
    index_difference = round(time_difference/time_step)
    transient_index = time_to_step_map(t_transient,time_step)
    stationary_trajectories = trajectories[:,1,transient_index:]

    if index_difference == 0:
        return np.mean(stationary_trajectories ** 2)

    initial_position = stationary_trajectories[:, :-index_difference]
    final_position = stationary_trajectories[:, index_difference:]

    return np.mean(initial_position*final_position)

def kl_divergence_gaussians(p,q):
    """KL(P || Q) for two multivariate Gaussians, closed form.
    Parameters:
    p,q: scipy.stats.like objects with atributes mean and cov

    Returns: float, kullback leibler divergence of p,q"""
    mean_p = p.mean
    mean_q = q.mean
    cov_p = p.cov
    cov_q = q.cov
    k = mean_p.shape[0]
    cov_q_inv = np.linalg.inv(cov_q)
    diff = mean_q - mean_p

    trace_term = np.trace(cov_q_inv @ cov_p)
    mean_term = diff @ cov_q_inv @ diff

    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    log_det_term = logdet_q - logdet_p

    return 0.5 * (trace_term + mean_term - k + log_det_term)

def symmetric_kl_gaussian(p,q):
    """Compute symmetric kullback leibler divergence of two distributions
    Parameters:
    p,q: scipy.stats.like objects with atributes mean and cov

    Returns: float, symmetric kullback leibler divergence of p,q"""
    return (kl_divergence_gaussians(p,q)+ kl_divergence_gaussians(q,p))/2

def analytical_distribution_eq(parameters, random_number_generator):
    """Compute the analytical distribution P(X,X_b) in equilibrium.

    Parameters:
    parameters: SystemVariables
    random_number_generator: numpy.random rng, to use as seed for comparability

    Reurns:
    scipy.stats.multivariat_normal representation of analytical P(X,X_b)"""
    means = np.array([0.0,0.0])
    kBT = parameters.k_B * parameters.T
    cov = np.array([[kBT/parameters.kappa, kBT/parameters.kappa],[kBT/parameters.kappa,kBT/parameters.kappa+kBT/parameters.kappa_b]])
    return stats.multivariate_normal(means,cov,random_number_generator)
             
def condition_gaussian(dist, fixed_indices, fixed_values):
    """condition a multivariate gaussian on a subset of the variables.

    Parameters:
    distribution: scipy.stats.multivariate_normal representing the joint
    fixed_indices: array of indices to condition on (the "b" block)
    fixed_values: values at which those indices are fixed
    
    returns: scipy.stats.multivariate_normal object for the conditional distribution of the remaining variables"""


    mean = np.asarray(dist.mean)
    cov = np.asarray(dist.cov)
    dim = mean.shape[0]

    fixed_indices = np.atleast_1d(fixed_indices)
    free_indices = np.array([i for i in range(dim) if i not in fixed_indices]) #all indices except the fixed ones

    mu_a = mean[free_indices]
    mu_c = mean[fixed_indices]

    #fancy indexing works as intended for 1d arrays, only returning the array at the indices, but for matrices, we have to use np.ix to get the matrix for the given indices 
    Sigma_aa = cov[np.ix_(free_indices, free_indices)]
    Sigma_ac = cov[np.ix_(free_indices, fixed_indices)]
    Sigma_cc = cov[np.ix_(fixed_indices, fixed_indices)]
    Sigma_cc_inv = np.linalg.inv(Sigma_cc)

    x_c = np.atleast_1d(fixed_values)

    mu_cond = mu_a + Sigma_ac @ Sigma_cc_inv @ (x_c - mu_c)
    Sigma_cond = Sigma_aa - Sigma_ac @ Sigma_cc_inv @ Sigma_ac.T

    return stats.multivariate_normal(mu_cond, Sigma_cond)


def mean_initial_positions_from_trajectories(formated_trajectories, params, n_outcomes=2, return_std=False):
    """
    n_outcomes=2 (default) reproduces the ORIGINAL binary behavior exactly,
    for full backward compatibility with existing pipeline code that calls
    this function without the n_outcomes argument.

    n_outcomes >= 3 uses the symmetric ±x_thresh scheme with the interior
    partitioned into n_outcomes - 2 equal-width bins.
    """
    row0 = formated_trajectories[0]  # X_0
    row2 = formated_trajectories[2]  # X_t
    row3 = formated_trajectories[3]  # X_bt

    edges = discretization_edges(params, n_outcomes)
    labels = outcome_labels(n_outcomes)

    labels0 = assign_outcome_labels(row0, edges, labels)
    labelst = assign_outcome_labels(row2, edges, labels)

    results = {}
    for l0 in labels:
        for lt in labels:
            filt = (labels0 == l0) & (labelst == lt)
            n = filt.sum()
            key = f"m_0={l0},m_t={lt}"
            results[key] = {
                "mean_X_t": row2[filt].mean() if n > 0 else np.nan,
                "mean_X_bt": row3[filt].mean() if n > 0 else np.nan,
                "mean_X_t_squared": np.mean(row2[filt]**2),
                "counts": n,
                
            }
            if return_std:
                results[key]["sigma_X_t"] = row2[filt].std() if n > 0 else np.nan
                results[key]["sigma_X_bt"] = row3[filt].std() if n > 0 else np.nan
    return results

def compute_c1_from_joint(m_0, m_t, params, joint_distribution, n_outcomes=2, n_sigma=8):
    """
    Compute c1 = E[X_t | m_0, m_t] (initial colloid position) from joint probability distribution.

    Any infinite integration bound (from `set_bounds`) is truncated to
    n_sigma standard deviations of the *fitted* marginal - using the
    distribution's own mean/covariance since the fitted Gaussian's actual scale is what
    determines whether scipy.integrate.dblquad's infinite-bound transform
    behaves well or underflows to zero.

    Parameters
    ----------
    m_0, m_t : int
        Measurement outcome labels.
    params : SystemVariables
    joint_distribution : scipy.stats.multivariate_normal-like
        joint distribution with a `.marginal(indices)` method
        (indices 0 = X_0, 2 = X_t in the full 4D fit).
    n_outcomes : int, optional
        Passed to `set_bounds`; default 2 (binary convention).
    n_sigma : float, optional
        Truncation width for infinite bounds; see `finite_bound`.

    Returns
    -------
    float
        c1 = E[X_t | m_0, m_t], in the same units as `joint_distribution`.
    """
    low_bound_0, up_bound_0 = set_bounds(params, m_0, n_outcomes)
    low_bound_t, up_bound_t = set_bounds(params, m_t, n_outcomes)

    marginal_X_0_X_t = joint_distribution.marginal([0, 2])
    mean_0, std_0 = marginal_X_0_X_t.mean[0], np.sqrt(marginal_X_0_X_t.cov[0, 0])
    mean_t, std_t = marginal_X_0_X_t.mean[1], np.sqrt(marginal_X_0_X_t.cov[1, 1])

    low_bound_0 = finite_bound(low_bound_0, mean_0, std_0, n_sigma)
    up_bound_0 = finite_bound(up_bound_0, mean_0, std_0, n_sigma)
    low_bound_t = finite_bound(low_bound_t, mean_t, std_t, n_sigma)
    up_bound_t = finite_bound(up_bound_t, mean_t, std_t, n_sigma)

    pdf = marginal_X_0_X_t.pdf
    norm_integral = dblquad(lambda X_0, X_t: pdf([X_0, X_t]),
                             low_bound_t, up_bound_t, lambda x: low_bound_0, lambda x: up_bound_0)
    weighting_integral = dblquad(lambda X_0, X_t: X_t * pdf([X_0, X_t]),
                                  low_bound_t, up_bound_t, lambda x: low_bound_0, lambda x: up_bound_0)

    return weighting_integral[0] / norm_integral[0]#quad functions return value and error, we want to divide values ->[0]


def compute_c3_from_joint(m_0, m_t, params, joint_distribution, n_outcomes=2, n_sigma=8):
    """
    Compute c3 (initial bath particle position) by integrating the marginal
    (X_0, X_t, X_bt) distribution of a joint Gaussian: X_0, X_t are
    restricted to the (m_0, m_t) outcome region, while X_bt ranges over its
    full support (not directly restricted by any measurement).

    All infinite integration bounds are truncated to n_sigma standard
    deviations of the corresponding *fitted* marginal component - see
    `compute_c1_from_joint` for the rationale.

    Parameters
    ----------
    m_0, m_t : int
        Measurement outcome labels.
    params : SystemVariables
    joint_distribution : scipy.stats.multivariate_normal-like
        Fitted joint distribution with a `.marginal(indices)` method
        (indices 0 = X_0, 2 = X_t, 3 = X_bt in the full 4D fit).
    n_outcomes : int, optional
        Passed to `set_bounds`; default 2 (binary convention).
    n_sigma : float, optional
        Truncation width for infinite bounds; see `finite_bound`.

    Returns
    -------
    float
        c3 = E[X_bt | m_0, m_t], in the same units as `joint_distribution`.
    """
    low_bound_0, up_bound_0 = set_bounds(params, m_0, n_outcomes)
    low_bound_t, up_bound_t = set_bounds(params, m_t, n_outcomes)

    marginal_X_0_X_t_X_bt = joint_distribution.marginal([0, 2, 3])
    marginal_X_0_X_t = joint_distribution.marginal([0, 2])

    mean_0, std_0 = marginal_X_0_X_t.mean[0], np.sqrt(marginal_X_0_X_t.cov[0, 0])
    mean_t, std_t = marginal_X_0_X_t.mean[1], np.sqrt(marginal_X_0_X_t.cov[1, 1])
    mean_bt, std_bt = marginal_X_0_X_t_X_bt.mean[2], np.sqrt(marginal_X_0_X_t_X_bt.cov[2, 2])

    low_bound_0 = finite_bound(low_bound_0, mean_0, std_0, n_sigma)
    up_bound_0 = finite_bound(up_bound_0, mean_0, std_0, n_sigma)
    low_bound_t = finite_bound(low_bound_t, mean_t, std_t, n_sigma)
    up_bound_t = finite_bound(up_bound_t, mean_t, std_t, n_sigma)
    low_bound_bt = finite_bound(-np.inf, mean_bt, std_bt, n_sigma)
    up_bound_bt = finite_bound(np.inf, mean_bt, std_bt, n_sigma)

    pdf_3d = marginal_X_0_X_t_X_bt.pdf
    pdf_2d = marginal_X_0_X_t.pdf

    numerator_integral = tplquad(
        lambda X_0, X_t, X_bt: X_bt * pdf_3d([X_0, X_t, X_bt]),
        low_bound_bt, up_bound_bt,
        lambda x: low_bound_t, lambda x: up_bound_t,
        lambda x, y: low_bound_0, lambda x, y: up_bound_0,
    )
    denominator_integral = dblquad(
        lambda X_0, X_t: pdf_2d([X_0, X_t]),
        low_bound_t, up_bound_t, lambda x: low_bound_0, lambda x: up_bound_0,
    )

    return numerator_integral[0] / denominator_integral[0]#quad functions return value and error, we want to divide values ->[0]


def compute_second_moment_Xt_from_joint(m_0, m_t, params, joint_distribution,n_outcomes=2):
    """Compute average X_t^2(initial colloid position) from joint probability distribution.

    Parameters:
    params: SystemVariables object
    m_0: integer value of bined first measurement 
    m_t: integer value of bined second measurement 
    joint_probability-distribution: scipy.stats.multivariate_normal

    Return:
    float, X_t^2"""
    low_bound_0, up_bound_0 = set_bounds(params, m_0,n_outcomes)
    low_bound_t, up_bound_t = set_bounds(params, m_t,n_outcomes)
    marginal_X_0_X_t = joint_distribution.marginal([0,2]).pdf
    norm_integral = dblquad(lambda X_0, X_t: marginal_X_0_X_t([X_0,X_t]),low_bound_t,up_bound_t, lambda x: low_bound_0, lambda x: up_bound_0)
    weigthing_integral = dblquad(lambda X_0, X_t: marginal_X_0_X_t([X_0,X_t]) * X_t**2,low_bound_t,up_bound_t, lambda x: low_bound_0, lambda x: up_bound_0)

    return weigthing_integral[0]/norm_integral[0] #quad functions return value and error, we want to divide values ->[0]

def mean_initial_positions_from_distributions(joint_distribution, params, n_outcomes=2):
    """
    Analytical counterpart to mean_initial_positions_from_trajectories: instead
    of binning empirical samples, integrates a fitted joint Gaussian directly
    via compute_c1_from_fit/compute_c3_from_fit. Returns the same dict shape,
    so it is a drop-in replacement anywhere mean_initial_positions_from_trajectories
    is used (e.g. as input to process_combination).

    Parameters:
    joint_distribution: scipy.stats.multivariate_normal, 4d fitted gaussian in X_),X_b0, X_t,X_bt
    params: SystemVariables object
    n_outcomes: integer, number of possible measurement outcomes
    """
    labels = outcome_labels(n_outcomes)
    results = {}

    for m_0 in labels:
        for m_t in labels:
            key = f"m_0={m_0},m_t={m_t}"
            try:
                mean_X_t = compute_c1_from_fit(m_0, m_t, params, joint_distribution, n_outcomes)
                mean_X_bt = compute_c3_from_fit(m_0, m_t, params, joint_distribution, n_outcomes)
                mean_X_t_squared, probability = compute_second_moment_xt_from_fit(
                    m_0, m_t, params, joint_distribution, n_outcomes
                )
            except ZeroDivisionError:
                # normalization integral was ~0: this outcome combination has
                # negligible probability mass under the fitted distribution
                mean_X_t, mean_X_bt, mean_X_t_squared, probability = np.nan, np.nan, np.nan, 0.0

            results[key] = {
                "mean_X_t": mean_X_t,
                "mean_X_bt": mean_X_bt,
                "mean_X_t_squared": mean_X_t_squared,
                "counts": None,   #no counts exist
            }

    return results

def bootstrap_initial_positions_from_trajectories(
    trajectories, t_second_measurement, time_step, params,random_number_generator=None,
    full_size=None, subsample_fraction=0.3, n_bootstrap=200,
    n_outcomes=2, with_replacement=False
):
    """
    Estimate mean initial conditions (per measurement outcome) via
    subsampling bootstrap: repeatedly draw a subset of a chosen-size pool
    of trajectories, compute the mean initial conditions on that subset

    Parameters: 
    trajectories : np.ndarray
        Full array of simulated trajectories, as returned by
        euler_maruyama_ensemble_bath_model.
    t_second_measurement : float
    time_step : float
        Output time step at which trajectories was saved.
    params : SystemVariables
    full_size : int, optional
        Number of trajectories (out of trajectories.shape[0]) to treat
        as the "full" pool for this bootstrap study. If None, uses all
        available trajectories.
    subsample_fraction : float, optional
        Fraction of full_size to draw in each bootstrap replicate.
        Default 0.3 (30%).
    n_bootstrap : int, optional
        Number of bootstrap replicates. Default 200; should be large
        enough that the replicate standard deviation itself is stable.
    n_outcomes : int, optional
        Default 2.
    random_number_generator : numpy.random.Generator, optional
        Used both to pick the fixed "full" pool (if full_size is not None)
        and to draw each bootstrap replicate.

    Returns:
    dict
        Keys "m_0={m_0},m_t={m_t}" (see `outcome_labels`), each mapping to
        a dict with:
            "mean_X_t", "mean_X_bt", "mean_X_t_squared" : float
                Mean across bootstrap replicates point 
            "std_X_t", "std_X_bt", "std_X_t_squared" : float
                Sample standard deviation (ddof=1) across bootstrap
                replicates   
    """

    total_available = trajectories.shape[0]
    if full_size is None:
        full_size = total_available

    # Fixed "full" pool for this study (reproducible across different
    # subsample_fraction values at the same full_size, given the same rng state).
    if full_size < total_available:
        pool_indices = random_number_generator.choice(total_available, size=full_size, replace=False)
        pool = trajectories[pool_indices]
    else:
        pool = trajectories

    subsample_size = max(1, round(subsample_fraction * full_size))

    labels = outcome_labels(n_outcomes)
    combos = [(m_0, m_t) for m_0 in labels for m_t in labels]
    replicate_values = {
        f"m_0={m_0},m_t={m_t}": {"mean_X_t": [], "mean_X_bt": [], "mean_X_t_squared": []}
        for m_0, m_t in combos
    }

    for _ in range(n_bootstrap):
        index = random_number_generator.choice(full_size, size=subsample_size, replace=False)
        subsample = pool[index]

        formatted = format_trajectories_at_two_times(subsample, 0, t_second_measurement, time_step)
        result = mean_initial_positions_from_trajectories(formatted, params, n_outcomes)

        for key, vals in result.items():
            replicate_values[key]["mean_X_t"].append(vals["mean_X_t"])
            replicate_values[key]["mean_X_bt"].append(vals["mean_X_bt"])
            replicate_values[key]["mean_X_t_squared"].append(vals["mean_X_t_squared"])

    summary = {}
    for key, vals in replicate_values.items():
        arr_t = np.array(vals["mean_X_t"], dtype=float)
        arr_bt = np.array(vals["mean_X_bt"], dtype=float)
        arr_t2 = np.array(vals["mean_X_t_squared"], dtype=float)

        #For small subsamples, there might be no realizations in a specific realization for a 
        #specific measurement outcome, leading to mean nan. ->use nanmean etc
        summary[key] = {
            "mean_X_t": np.nanmean(arr_t),
            "std_X_t": np.nanstd(arr_t, ddof=1),
            "mean_X_bt": np.nanmean(arr_bt),
            "std_X_bt": np.nanstd(arr_bt, ddof=1),
            "mean_X_t_squared": np.nanmean(arr_t2),
            "std_X_t_squared": np.nanstd(arr_t2, ddof=1),
        }

    return summary


def make_trajectory_initial_position_estimator(trajectories, time_step):
    """
    Adapt `mean_initial_positions_from_trajectories` to the
    ``(params, t_second_measurement, n_outcomes)`` calling convention
    expected by `get_or_compute_initial_positions`.

    The returned estimator slices the pre-simulated `trajectories` array
    at times ``(0, t_second_measurement)`` 

    Parameters
    ----------
    trajectories : numpy.ndarray
        Simulated trajectories as returned by
        `euler_maruyama_ensemble_bath_model`, with shape
        ``(number_of_samples, dimension + 1, number_of_saves + 1)``.
    time_step : float
        The output time step at which `trajectories` was saved (i.e. the
        resolution passed to the integrator as ``output_time_step``).

    Returns
    -------
    callable
        A function ``estimate_initial_positions(params,
        t_second_measurement, n_outcomes=2)`` returning the same dict
        structure as `mean_initial_positions_from_trajectories`.
    """
    def estimate_initial_positions(params, t_second_measurement, n_outcomes=2):
        formatted_trajectories = format_trajectories_at_two_times(
            trajectories, 0, t_second_measurement, time_step
        )
        return mean_initial_positions_from_trajectories(
            formatted_trajectories, params, n_outcomes
        )

    return estimate_initial_positions


def make_distribution_initial_position_estimator(
    trajectories, time_step, random_number_generator=None,
):
    """
    Build a distribution-based initial-position estimator that fits a new distribution from `trajectories`
    for any `t_second_measurement` 

    Parameters
    ----------
    trajectories : numpy.ndarray
        Simulated trajectories as returned by
        `euler_maruyama_ensemble_bath_model`, used as the fallback data
        source for fitting a joint Gaussian at any
        `t_second_measurement` missing from the cache.
    time_step : float
        The output time step at which `trajectories` was saved; used to
        slice it via `format_trajectories_at_two_times`.
    random_number_generator : numpy.random.Generator, optional
        Passed through to `fit_gaussian` as the seed of the resulting
        ``scipy.stats.multivariate_normal`` (only relevant if samples are
        later drawn from it). Default None.

    Returns
    -------
    callable
        A function ``estimate_initial_positions(params,
        t_second_measurement, n_outcomes=2)`` satisfying the same calling
        convention as `mean_initial_positions_from_analytical`, suitable
        for passing to `get_or_compute_initial_positions`.
    """

    def estimate_initial_positions(params, t_second_measurement, n_outcomes=2):
        
        formatted_trajectories = format_trajectories_at_two_times(trajectories, 0, t_second_measurement, time_step)
        joint_distribution = fit_gaussian(formatted_trajectories, random_number_generator)
        return mean_initial_positions_from_distributions(joint_distribution, params, n_outcomes)

    return estimate_initial_positions
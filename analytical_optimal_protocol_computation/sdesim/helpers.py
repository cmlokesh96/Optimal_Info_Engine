"""This file contains functions which mainly exist to support functionalities of others, following the DRY philosophy"""
import numpy as np
import os
from scipy import stats
from scipy.interpolate import interp1d
import csv

def time_to_step_map(t,timestep):
    """Compute index of a given time in the simulation in the result array"""
    return round(t/timestep)

def ensure_parent_dir(path):
    """
    Create the parent directory of path if it doesn't already exist.

    Parameters:
    path : str
        Full file path (not directory) about to be written to.
    """
    parent = os.path.dirname(path)
    if parent:  # empty string if path has no directory component
        os.makedirs(parent, exist_ok=True)

#functions to save and load mean and covariance of Gaussians so they can be im/exported to other scripts
def save_fitted_distribution(fitted_distribution, save_path):
    """Save parameters of a multivariate gaussian in npz format.

    Parameters:
    fitted_distribution: scipy.stats.multivariate_normal
    save_path: location to save parameters to"""
    ensure_parent_dir(save_path) 
    np.savez(save_path, mean=fitted_distribution.mean, cov=fitted_distribution.cov)

def load_fitted_distribution(save_path, random_number_generator=None):
    """load distribution from a npz file as created by save_fitted_distribution

    Parameters:
    save_path: location to load parameters from

    Return: scipy.stats.multivariate_normal with mean and std from file"""
    data = np.load(save_path)
    return stats.multivariate_normal(data['mean'], data['cov'], seed=random_number_generator)

def format_trajectories_at_time(trajectories, t, time_step, variable_indices=None):
    """slice X and X_b at a specific time from the trajectory array and reshape so that all X values form a row and those of X_b as well
    
    Parameters:
    trajectories: np.ndarray as returned by the integrators from the integrator module
    t: float, time at which the values of the variables are extracted
    time_step: float, time resolution of trajectories 
    variable indices is an array containing the indices of the variables that are to keep
    
    Returns: aarray with a row with variable values at given time for each variable"""
    index = time_to_step_map(t, time_step)
    if variable_indices is None:
        variable_indices = np.arange(1, trajectories.shape[1])  # indices of the non time rows, fall back in case no others given

    data = trajectories[:, variable_indices, index]  # shape (number_of_samples, num_variables), only one time index reduces dimensionality

    return data.T  # shape (num_variables, number_of_samples)

def format_trajectories_at_two_times(trajectories, t1, t2, time_step, variable_indices=np.array([1, 2])):
    """slice X and X_b at two given times from the trajectory array and reshape so that all X values form a row and those of X_b as well
    
    Parameters:
    trajectories: np.ndarray as returned by the integrators from the integrator module
    t1, t2: float, times at which the values of the variables are extracted
    time_step: float, time resolution of trajectories 
    variable indices is an array containing the indices of the variables that are to keep
    
    Returns: aarray with a row with variable values at given times for each variable, first all variables at the first time, then those at the second"""
    index1 = time_to_step_map(t1, time_step)
    index2 = time_to_step_map(t2, time_step)

    data_t1 = trajectories[:, variable_indices, index1]   # shape (num_samples, 2)
    data_t2 = trajectories[:, variable_indices, index2]    # shape (num_samples, 2)

    return np.vstack((data_t1.T, data_t2.T))               # shape (4, num_samples)

#Does the same as format_trajectories_at_time just that all samples from a transient time on are included

def format_trajectories_stationary(trajectories, t_transient, time_step, variable_indices=None):
    """return array with row for each variable containing all values after a transient time
    
    Parameters:
    trajectories: np.ndarray as returned by the integrators from the integrator module
    t_transient: float, time at which the process is assumed to be stationary
    time_step: float, time resolution of trajectories 
    variable indices is an array containing the indices of the variables that are to keep
    
    Returns: aarray with a row with variable values after given time for each variable"""
    transient_index = time_to_step_map(t_transient, time_step)
    if variable_indices is None:
        variable_indices = np.arange(1, trajectories.shape[1])

    data = trajectories[:, variable_indices, transient_index:]        # (samples, variables, steps)
    data = np.moveaxis(data, 1, 0)                                    # (variables, samples, steps)
    num_variables = data.shape[0]
    return data.reshape(num_variables, -1)                            # (variables, samples*steps)

#To enable simulations with the same initial condition for all runs or different ones drawn according to an distribution, the euler-maruyama
#expects an array of initial conditions for the trajectories which can be generated with these two functions:
def draw_initial_conditions(fitted_distribution, number_of_samples, random_number_generator):
    """Draw initial conditions for all trajectories simulated at once from the distribution

    Parameters:
    fitted_distribution: scipy.stats.multivariate_gaussian, distribution to draw from
    number_of_sampes: int, number of initial conditions to generate
    random_number_generator: np.random rng, already seeded, used for comparability of initila conditions between runs.

    Returns: array of arrays of initial_conditions"""
    initial_states = fitted_distribution.rvs(size=number_of_samples, random_state=random_number_generator)
    #For an 1 dimensional gaussian, the output would be an  1d array whilst for further use, we need a 2d array, so we reshape in this case
    if initial_states.ndim == 1:
        initial_states = initial_states.reshape(-1, 1)
    return initial_states

def reshape_constant_initial_state(initial_state, number_of_samples):
    """repeat same initial condition in array for use in integrators:

    Parameters: array like, initial condition to be repeated
    number_of_samples: int, how often initial condition is to be repeated in array

    Return: np. array containing initial_condition number_of_samples times"""
    return np.tile(initial_state, (number_of_samples, 1))#The same initial condition is repeated number of samples times

def parse_m_values(string):
    """infer values of m_0, m_t from string, for instance filenames containing m_0/t=...

    Parameters:
    string: str, to infer from

    Return:  np.array containin value of m_0, m_t"""
    # Split the string into individual "key=value" pairs
    pairs = string.split(',')

    # Build a dictionary {key: value} by splitting each pair on "="
    values = {}
    for pair in pairs:
        key, value = pair.split('=')
        values[key.strip()] = float(value.strip())

    return np.array([values['m_0'], values['m_t']])

def compute_optimal_trajectories_array(B_creator, t_f, c_vector, lambda_f):
    """compute X, X_b and lambda at 1000 times during the execution of the protocol

    Parameters:

    B_creator: SolutionMatrixCreator
    t_f: float, end time of protocol
    c_vector: np.array, initial_condition vector
    lambda_f: float, final position of trap

    Return:
    np.ndarray containing times, X, X_b and lambda values as rows"""

    
    resolution = t_f / 1000
    final_index = time_to_step_map(t_f, resolution)
    optimal_trajectories = np.empty((4, final_index + 1))
    for i in range(final_index + 1):
        t = resolution * i
        current_B = B_creator.current_matrix(t)
        current_solution = current_B @ c_vector
        optim_lambda = (B_creator.tau_p * current_solution[1]
                         + (B_creator.k + 1) * current_solution[0]
                         - current_solution[2]) / B_creator.k

        optimal_trajectories[0, i] = t
        optimal_trajectories[1, i] = current_solution[0]
        optimal_trajectories[2, i] = current_solution[2]
        optimal_trajectories[3, i] = optim_lambda
        if i == final_index:
            optimal_trajectories[3, i] = lambda_f

    return optimal_trajectories

def compute_work_per_step(params, optimal_trajectories):
    """compute work that is done in each step and the accumulated work at each step

    Parameters:
    params: SystemVariables
    optimal_trajectories: np.ndarray as returned by compute_optimal_trajectories_array

    Returns:
    np.ndarray containing rows for time, change of work in between and accumulated work"""

    number_of_timesteps = optimal_trajectories.shape[1]
    results = np.empty((3,number_of_timesteps))

    results[0,:] = optimal_trajectories[0,:]
    # Step 0 is the initial jump itself: trap held at rest (lambda=0) before
    # this decision, then jumps straight to optimal_trajectories[3,0] (which
    # may be nonzero - see compute_optimal_trajectories_array) while the
    # particle sits at optimal_trajectories[1,0]. Same work formula as every
    # other step below, just against an implicit lambda=0 starting point
    # instead of the previous array entry - omitting this (treating step 0
    # as free) would silently drop the jump's real energy cost from the
    # accumulated total.
    X0, lambda0 = optimal_trajectories[1, 0], optimal_trajectories[3, 0]
    results[1,0] = 0.5*params.kappa * ((lambda0 - X0)**2 - (0.0 - X0)**2)
    results[2,0] = results[1,0]
    for i in range(1,number_of_timesteps):
        results[1,i] = 0.5*params.kappa *((optimal_trajectories[3,i]-optimal_trajectories[1,i-1])**2 - (optimal_trajectories[3,i-1]-optimal_trajectories[1,i-1])**2)
        results[2,i] = results[2,i-1]+ results[1,i]

    return results

def _fmt(x):
    """Format a float nicely for use in a filename (no dots, no trailing zeros)."""
    return f"{x:g}".replace(".", "p").replace("-", "neg")


def make_save_path(base_dir, t_second_measurement, t_protocol_end, m_0, m_t, extension="npz"):
    """Create a save_path containing specifications about characteristic times and measuremnet outcomes
    
    Parameters:
    base_dir: string, directory path to save to
    t_second_measurement, t_protocol_end: float
    m_0,m_t: int, outcomes of measurements at times 0 and t
    returns; joint_path using os (safer then simple string)"""
    os.makedirs(base_dir, exist_ok=True)#crate dir if non existent
    filename = (
        f"t2_{_fmt(t_second_measurement)}_"
        f"tf_{_fmt(t_protocol_end)}_"
        f"m0_{int(m_0)}_mt_{int(m_t)}.{extension}"
    )
    return os.path.join(base_dir, filename)

def load_optimal_protocol(save_dir, t_second_measurement, t_protocol_end, m_0, m_t):
    """
    Load previously saved optimal protocol data for a given parameter combination.

    Returns a dict with keys:
        'optimal_trajectories', 'mean_work', 'c_vector', 'lambda_f',
        't_second_measurement', 't_protocol_end', 'm_0', 'm_t'
    """
    load_path = make_save_path(save_dir, t_second_measurement, t_protocol_end, m_0, m_t)
    data = np.load(load_path)

    return {
        "optimal_trajectories": data["optimal_trajectories"],
        "mean_work": data["mean_work"].item(),
        "c_vector": data["c_vector"],
        "lambda_f": data["lambda_f"].item(),
        "t_second_measurement": data["t_second_measurement"].item(),
        "t_protocol_end": data["t_protocol_end"].item(),
        "m_0": data["m_0"].item(),
        "m_t": data["m_t"].item(),
    }

def optimize_quadratic_form(D, fixed_indices, fixed_values, elim_indices):
    """
    Direct solve of the stationarity condition for W(c) = c^T D c,
    given fixed c-values at `fixed_indices` and optimizing over `elim_indices`.
    Indices are 0-based positions into D (c1 -> 0, c2 -> 1, ..., c5 -> 4).
    """
    D = np.asarray(D, dtype=float)
    fixed_indices = np.atleast_1d(fixed_indices)
    elim_indices = np.atleast_1d(elim_indices)
    c_fixed = np.atleast_1d(fixed_values)

    D_ee = D[np.ix_(elim_indices, elim_indices)]
    D_ef = D[np.ix_(elim_indices, fixed_indices)]

    return np.linalg.solve(D_ee, -D_ef @ c_fixed)


def discretization_edges(params, n_outcomes):
    """
    Build bin edges for discretizing a continuous coordinate into
    n_outcomes ordered categories.

    n_outcomes == 2: reproduces the ORIGINAL behavior exactly —
        a single threshold at x_thresh, splitting into
        m=0 (x < x_thresh) and m=1 (x >= x_thresh).
        (Note: -x_thresh is NOT used here, matching the legacy code.)

    n_outcomes >= 3: symmetric thresholds at -x_thresh and +x_thresh,
        with the interior [-x_thresh, x_thresh) split into
        (n_outcomes - 2) equal-width bins.
    """
    if n_outcomes < 2:
        raise ValueError("Need at least two outcomes (n_outcomes >= 2).")

    if n_outcomes == 2:
        return np.array([-np.inf, params.x_thresh, np.inf])

    inner_edges = np.linspace(-params.x_thresh, params.x_thresh, n_outcomes - 1)
    return np.concatenate(([-np.inf], inner_edges, [np.inf]))


def outcome_labels(n_outcomes):
    """ Create labels for the bins for general number of outcomes.

    Parameters:
    n_outcomes: number of bins to divide positions in

    Return:
    n_outcomes == 2: [0, 1]  
    n_outcomes == 3: [-1, 0, 1]
    n_outcomes == 5: [-2, -1, 0, 1, 2]
    n_outcomes == 4 (even, no exact center): [-2, -1, 0, 1]
    """
    if n_outcomes == 2:
        return [0, 1]
    if n_outcomes % 2 == 1:
        half = n_outcomes // 2
        return list(range(-half, half + 1))
    else:
        half = n_outcomes // 2
        return list(range(-half, half))


def assign_outcome_labels(x, edges, labels):
    """Map a continuous array x to its discrete outcome label."""
    bin_indices = np.digitize(x, edges[1:-1], right=False)
    return np.asarray(labels)[bin_indices]

def set_bounds(params, m, n_outcomes=2):
    """
    Return (low, high) integration bounds for the region corresponding to
    outcome label `m`, using the same discretization scheme as
    discretization_edges/outcome_labels (so analytical integration and
    empirical binning always agree on region boundaries).

    n_outcomes=2 reproduces the original binary convention exactly:
        m=0 -> (-inf, x_thresh),  m=1 -> (x_thresh, inf)
    n_outcomes>=3 uses the symmetric multi-bin scheme with labels from
    outcome_labels(n_outcomes).
    """
    edges = discretization_edges(params, n_outcomes)
    labels = outcome_labels(n_outcomes)

    if m not in labels:
        raise ValueError(f"m={m!r} is not a valid outcome for n_outcomes={n_outcomes}; "
                          f"valid labels are {labels}")

    index = labels.index(m)
    return edges[index], edges[index + 1]


def finite_bound(bound, mean, std, n_sigma=8):
    """
    Replace an infinite integration bound with a finite one, n_sigma
    standard deviations from `mean`.

    For a Gaussian(-like) integrand, the truncated tail beyond n_sigma=8
    contributes a negligible amount, but avoids
    scipy.integrate.quad/dblquad/tplquad's costly and numerically fragile
    infinite-interval transform, which can otherwise entirely miss a peak
    that is narrow relative to the nominal (untransformed) integration
    domain - this is exactly the failure mode that produces a norm
    integral of 0.0 (and a downstream ZeroDivisionError) when the
    integrand's actual support is many orders of magnitude narrower than
    where the adaptive sampler initially probes.

    Parameters
    ----------
    bound : float
        Original integration bound; may be +/- np.inf.
    mean : float
        Mean of the (approximately Gaussian) integrand along this axis.
    std : float
        Standard deviation of the integrand along this axis.
    n_sigma : float, optional
        Truncation width, in standard deviations. Default 8.

    Returns
    -------
    float
        `bound` unchanged if already finite; otherwise `mean + n_sigma*std`
        (for +inf) or `mean - n_sigma*std` (for -inf).
    """
    if np.isfinite(bound):
        return bound
    return mean + n_sigma * std if bound > 0 else mean - n_sigma * std

def make_python_ic_path(base_dir, t_second_measurement, method_name, params =None):
    """
    Construct the path to a Python-exported initial-conditions CSV
    (produced by `export_initial_conditions_for_matlab`), for a given
    (t_second_measurement). Unlike protocol files,
    a single initial-conditions file covers all (m_0, m_t) outcome
    combinations for that time pair (one row per combination).
    params allows to incorporate x_thresh and kappa in the file name

    Parameters
    ----------
    base_dir : str
    t_second_measurement : float

    Returns
    -------
    str
    """
    if params is not None:
        fname = (f"t2_{t_second_measurement:g}_kappa_{params.kappa:g}_"
                 f"xthresh_{params.x_thresh:g}_{method_name}_initial_conditions.csv")
    else:
        fname = f"t2_{t_second_measurement:g}_{method_name}_initial_conditions.csv"
    return os.path.join(base_dir, fname)

def export_initial_conditions(mean_initial_positions, t_second_measurement, base_dir, method_name, params=None):
    """
    Export conditional initial positions to a CSV readable by MATLAB.

    Parameters:
    mean_initial_positions : dict
    t_second_measurement : float
    base_dir : str
    method_name : str
    params : SystemVariables, optional
        Forwarded to make_python_ic_path.

    Return: str, Path that has been saved to
    """
    path = make_python_ic_path(base_dir, t_second_measurement, method_name, params)
    os.makedirs(base_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_second_measurement", "m_0", "m_t", "mean_X_t", "mean_X_bt", "mean_X_t_squared"])
        for key, vals in mean_initial_positions.items():
            m_0, m_t = parse_m_values(key)
            if np.isnan(vals["mean_X_t"]):
                continue
            writer.writerow([t_second_measurement, int(m_0), int(m_t),
                              vals["mean_X_t"], vals["mean_X_bt"], vals["mean_X_t_squared"]])
    return path


def make_matlab_protocol_path(base_dir, t_second_measurement, t_protocol_end, m_0, m_t):
    """
    Construct the path to a MATLAB-exported protocol CSV, matching the
    naming convention used by MATLAB's makeMatlabProtocolPath.m.

    Parameters:
    base_dir : str
    t_second_measurement, t_protocol_end : float
    m_0, m_t : int

    Returns:
    str
    """
    fname = f"t2_{t_second_measurement:g}_tf_{t_protocol_end:g}_m0_{int(m_0)}_mt_{int(m_t)}.csv"
    return os.path.join(base_dir, fname)


def load_matlab_protocol(path):
    """
    Load a MATLAB-exported (t, lambda) protocol CSV and return an
    interpolated callable lambda_func(t) suitable for
    `compute_work_general_protocol`.

    Parameters:
    path : str

    Returns:
    callable
        lambda_func(t) -> float, linearly interpolated from the exported
        (t, lambda) points. Evaluating outside the exported time range
        raises (no extrapolation), to avoid silently evaluating work
        beyond the range MATLAB actually trained/exported.
    """
    t_vals, lambda_vals = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_vals.append(float(row["t"]))
            lambda_vals.append(float(row["lambda"]))
    t_vals = np.asarray(t_vals)
    lambda_vals = np.asarray(lambda_vals)
    #interpolate discrete protocol values to continous function for evaluation
    interpolator = interp1d(t_vals, lambda_vals, kind="linear",
                             bounds_error=True, assume_sorted=True)

    def lambda_func(t):
        return float(interpolator(t))

    return lambda_func

def load_initial_conditions_from_csv(path):
    """
    Load a previously exported initial-conditions CSV (as written by
    `export_initial_conditions_for_matlab`) back into the same dict
    structure returned by `mean_initial_positions_from_analytical`/
    `mean_initial_positions_from_trajectories`, so a cached export can be
    used as a drop-in substitute for recomputing.

    Note: since export_initial_conditions_for_matlab skips NaN (negligible
    probability) outcomes when writing, any such outcome will simply be
    absent from the returned dict rather than present with "counts": None
    as the from-scratch analytical function would produce. Callers that
    rely on every outcome key being present (rather than checking `.get`
    or membership) should account for this.

    Parameters:
    path : str

    Returns:
    dict
        Keys "m_0={m_0},m_t={m_t}", values dicts with "mean_X_t",
        "mean_X_bt" (no "mean_X_t_squared"/"counts" - not currently
        exported; see export_initial_conditions_for_matlab).
    """
    positions = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m_0, m_t = int(float(row["m_0"])), int(float(row["m_t"]))
            key = f"m_0={m_0},m_t={m_t}"
            positions[key] = {
                "mean_X_t": float(row["mean_X_t"]),
                "mean_X_bt": float(row["mean_X_bt"]),
                "mean_X_t_squared": float(row["mean_X_t_squared"]),
            }
    return positions

def get_or_compute_initial_positions(params, t_second_measurement, base_dir, initial_condition_function, method_name,
                                      n_outcomes=2, force_recompute=False):
    """
    Load a cached initial-conditions CSV for (t_second_measurement,
    t_protocol_end) if one already exists at the path
    `make_python_ic_path` would produce; otherwise compute it via
    `mean_initial_positions_from_analytical` and write it to that path
    for future reuse.

    Parameters:
    params : SystemVariables
    t_second_measurement: float
    base_dir : str
        Directory to check/write the cached CSV in.
    initial_condition_function: function to compute initial conditions, either mean_initial_positions_from_analytical, mean_initial_positions_from_distributions or mean_initial_positions_from_trajectories
    n_outcomes : int, optional
        Default 2.
    force_recompute : bool, optional
        If True, always recompute and overwrite the cache, ignoring any
        existing file. Default False.

    Returns:
    dict
        As returned by mean_initial_positions_from_analytical or load_initial_conditions_from_csv (if loaded from
        cache) - see load_initial_conditions_from_csv's docstring for the
        one structural difference between these two return paths
        (absence vs. presence of NaN/negligible-probability entries).
    """
    path = make_python_ic_path(base_dir, t_second_measurement, method_name, params)

    if not force_recompute and os.path.exists(path):
        print(f"  Loading cached initial conditions ({method_name}) from {path}")
        return load_initial_conditions_from_csv(path)

    print(f"  Computing initial conditions via '{method_name}' (t2={t_second_measurement}, "
          f"kappa={params.kappa:g}, x_thresh={params.x_thresh:g})")
    positions = initial_condition_function(params, t_second_measurement, n_outcomes)
    export_initial_conditions(positions, t_second_measurement, base_dir, method_name, params)
    return positions

def scale_gaussian(dist, scale, seed=None):
    """Return a new scipy.stats.multivariate_normal representing Y = scale*X
    for X ~ dist. lets you correctly overlay a fitted Gaussian's pdf on
    an axis that has been rescaled for display 
    
    Parameters
    ----------
    dist : scipy.stats.multivariate_normal (frozen)
        Original distribution, with attributes `.mean` and `.cov`, in the
        original units (e.g. meters).
    scale : float
        Multiplicative factor converting original units to display units
        (e.g. 1e9 for meters -> nanometers).
    seed : int or numpy.random.Generator, optional
        Passed through to the resulting `scipy.stats.multivariate_normal`
        as its `seed`, only relevant if samples are later drawn from it.
        Default None.

    Returns
    -------
    scipy.stats.multivariate_normal (frozen)
        Distribution of Y = scale * X, with mean `dist.mean * scale` and
        covariance `dist.cov * scale**2`.
    """
    mean = np.atleast_1d(dist.mean) * scale
    cov = np.atleast_2d(dist.cov) * scale**2
    return stats.multivariate_normal(mean, cov, seed=seed)


def add_suptitle_and_legend(fig, title, handles=None, max_legend_cols=5, gap_in=0.15,
                              hspace=0.45, wspace=0.3):
    """
    Add a suptitle, and optionally a single shared legend, above a grid of
    subplots, then shrink the subplot grid so
    that both (if a legend is present) or just the title fit
    above any subplot titles, with a gap.

    Parameters:
   
    fig : matplotlib.figure.Figure
    title : str
    handles : list of matplotlib.lines.Line2D, optional
        Legend handles (with labels already set). If None or empty, no
        legend is created at all. Default None.
    max_legend_cols : int, optional
        Default 5.
    gap_in : float, optional
        Vertical gap, in inches, kept between the lowest header element
        and all subplot titles. Default 0.15.
    hspace, wspace : float, optional
        Passed through to fig.subplots_adjust. Defaults 0.45, 0.3.

    Returns
    -------
    None
    """
    has_legend = handles is not None and len(handles) > 0

    fig.suptitle(title, y=0.98)

    legend_artist = None
    if has_legend:
        legend_ncol = max(1, min(len(handles), max_legend_cols))
        legend_artist = fig.legend(handles=handles, loc='upper center',
                                   ncol=legend_ncol, bbox_to_anchor=(0.5, 0.93),
                                   frameon=True)

    fig.canvas.draw()  # force a real layout pass so extents below are accurate
    renderer = fig.canvas.get_renderer()
    fig_height_px = fig.bbox.height
    inv = fig.transFigure.inverted()

    if has_legend:
        # Bottom edge of the legend, in figure-fraction coordinates.
        legend_bbox = legend_artist.get_window_extent(renderer)
        header_bottom_fig = inv.transform((0, legend_bbox.y0))[1]
    else:
        # No legend: use the suptitle's own bottom edge instead.
        title_bbox = fig._suptitle.get_window_extent(renderer)
        header_bottom_fig = inv.transform((0, title_bbox.y0))[1]

    # How far does a subplot title extend above its own axes box? Same for
    # every subplot (same fontsize/pad), so one sample suffices.
    title_overshoot_fig = 0.0
    for ax in fig.axes:
        if ax.get_title():
            title_bbox = ax.title.get_window_extent(renderer)
            ax_bbox = ax.get_window_extent(renderer)
            overshoot_px = max(0.0, title_bbox.y1 - ax_bbox.y1)
            title_overshoot_fig = overshoot_px / fig_height_px
            break

    fig_height_in = fig.get_size_inches()[1]
    gap_fraction = gap_in / fig_height_in
    top_fraction = max(0.35, header_bottom_fig - gap_fraction - title_overshoot_fig)

    fig.subplots_adjust(top=top_fraction, hspace=hspace, wspace=wspace)


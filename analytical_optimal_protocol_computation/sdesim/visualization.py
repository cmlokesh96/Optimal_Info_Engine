"""Plot and visualize results"""
import shutil
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from .statistics import time_to_step_map, fit_gaussian, bootstrap_initial_positions_from_trajectories
from .statistics import stationary_correlation, correlation_to_init_time, OUP_stationary_autocorrrelation  
import numpy as np
from .helpers import format_trajectories_at_time, format_trajectories_stationary, scale_gaussian, add_suptitle_and_legend, outcome_labels, load_optimal_protocol, make_matlab_protocol_path, load_matlab_protocol, compute_work_per_step,ensure_parent_dir
from .constants import POSITION_SCALE, FINE_TIME_STEP_NON_EQ
from sdesim.integrators import simulate_stochastic_trajectories_under_optimal_protocol



def setup_matplotlib():
    """setup latex specific plotting parameters"""
    # Use real LaTeX for text rendering when available (best typesetting);
    # fall back to matplotlib's built-in mathtext otherwise so plotting
    # doesn't crash on machines without a LaTeX install.
    plt.rcParams['text.usetex'] = shutil.which('latex') is not None
    plt.rcParams['text.latex.preamble'] = r'\usepackage{xfrac, amsmath}'
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 23
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.titlesize'] = 30 
    plt.rcParams['figure.titlesize'] = 30 
    plt.rcParams['errorbar.capsize'] = 5

def create_fig(figsize=(12, 7.5)):
    """return: fig, ax object with unified layout"""
    fig, axes = plt.subplots(figsize=figsize)
    axes.tick_params(axis='both', labelsize='large')  # or remove per above
    axes.grid(which='major', color='#CCCCCC', linestyle='--')
    axes.grid(which='minor', color='#CCCCCC', linestyle=':')
    axes.xaxis.set_minor_locator(AutoMinorLocator(5))
    axes.yaxis.set_minor_locator(AutoMinorLocator(5))
    axes.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3), useMathText=True)
    return fig, axes



def plot_histogramm(t, timestep, trajectories, number_of_bins, save_path, random_number_generator, title, variable_index=1):
    """plot a histogram at a specific time, samples from multiple trajectories, fit a gaussian to that distribution.

    Parameters:
    t: float: time at which the distribution is plotted
    timestep: float, time resolution of trajectories 
    number_of_bins: int, number of bins to divide sample space in
    save_path: str, path to save figure to
    random_number_generator: np.random rng, used to seed fitted distribution for comparability
    title: str, title of the plot
    variable_index: int, optional: row of trajectories of which histogram is to be plotted, default 1(X)

    Returns:
    scipy.stats.multivariate_gaussian obtained by fit to data"""
    index = time_to_step_map(t,timestep)
    counts, bin_edges = np.histogram(trajectories[:, variable_index, index]*POSITION_SCALE, bins=number_of_bins, density=True)
    fitted_distribution = fit_gaussian(format_trajectories_at_time(trajectories,t, timestep, variable_index),random_number_generator)
    plot_dist = scale_gaussian(fitted_distribution, POSITION_SCALE)
    fig, ax = create_fig()
    variable_names = ["$t$", "$X$", "$X_{\\mathrm{b}}"]
    ax.set_xlabel(variable_names[variable_index] + "in nm")
    ax.set_ylabel("Samples in bin")
    ax.set_title(title)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1]-bin_edges[0]
    ax.bar(bin_centers, counts, bin_width, align='center', fill=True)
    plotting_range = np.linspace(bin_centers[0],bin_edges[-1],700)
    ax.plot(plotting_range, plot_dist.pdf(plotting_range),c='red')
    ensure_parent_dir(save_path) 
    plt.savefig(save_path)
    plt.close(fig)
    return fitted_distribution

def plot_histogramm_stationary(t_transient, timestep, trajectories, number_of_bins, save_path, random_number_generator, title, variable_index = 1):
    """plot a histogram in stationary state, samples from multiple trajectories and times in stationary state, fit a gaussian to that distribution.

    Parameters:
    t_transient: float: time at which the system is assumed to have reached a stationary state
    timestep: float, time resolution of trajectories 
    number_of_bins: int, number of bins to divide sample space in
    save_path: str, path to save figure to
    random_number_generator: np.random rng, used to seed fitted distribution for comparability
    title: str, title of the plot
    variable_index: int, optional: row of trajectories of which histogram is to be plotted, default 1(X)

    Returns:
    scipy.stats.multivariate_gaussian obtained by fit to data"""
    
    transient_index = time_to_step_map(t_transient,timestep)
    positions_stationary = trajectories[:, variable_index, transient_index:]*POSITION_SCALE
    fitted_distribution = fit_gaussian(format_trajectories_stationary(trajectories,t_transient, timestep, variable_index),random_number_generator)
    plot_dist = scale_gaussian(fitted_distribution, POSITION_SCALE)
    counts, bin_edges = np.histogram(positions_stationary.flatten(), bins=number_of_bins, density=True)
    fig, ax = create_fig()
    variable_names = ["$t$", "$X$", "$X_{\\mathrm{b}}"]

    ax.set_xlabel(variable_names[variable_index] + "in nm")
    ax.set_ylabel("Samples in bin")
    ax.set_title(title)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1]-bin_edges[0]
    ax.bar(bin_centers, counts, bin_width, align='center', fill=True)
    plotting_range = np.linspace(bin_centers[0],bin_edges[-1],700)
    ax.plot(plotting_range, plot_dist.pdf(plotting_range),c='red')
    ensure_parent_dir(save_path) 
    plt.savefig(save_path)
    plt.close(fig)
    return fitted_distribution

def plot_heatmap(t, timestep, trajectories, number_of_bins, save_path, variable_indices=np.array([1,2]), random_number_generator= None):
    """plot a heatmap of two random variables at a specific time, samples from multiple trajectories, fit a gaussian to that distribution.

    Note: assumes first variable to be X second to be X_b in axis labels, rework for other use cases

    Parameters:
    t_transient: float: time at which the distribution is plotted
    timestep: float, time resolution of trajectories 
    number_of_bins: int, number of bins to divide sample space in
    save_path: str, path to save figure to
    random_number_generator: np.random rng, used to seed fitted distribution for comparability
    title: str, title of the plot
    variable_indices: array, optional: rows of trajectories of which histogram is to be plotted, default [1,2} (X,X_b)

    Returns:
    scipy.stats.multivariate_gaussian obtained by fit to data"""
    index = time_to_step_map(t, timestep)
    x_at_t = trajectories[:, variable_indices[0], index]*POSITION_SCALE
    xb_at_t = trajectories[:, variable_indices[1], index]*POSITION_SCALE
    fitted_distribution = fit_gaussian(format_trajectories_at_time(trajectories,t, timestep, variable_indices),random_number_generator)
    plot_dist = scale_gaussian(fitted_distribution, POSITION_SCALE)
    counts, xedges, yedges = np.histogram2d(
        x_at_t, xb_at_t, bins=number_of_bins, density=True
    )

    fig, ax = create_fig()
    
    heatmap_mesh = ax.pcolormesh(xedges, yedges, counts.T, shading='auto')

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    grid_points = np.dstack((X, Y))
    pdf_values = plot_dist.pdf(grid_points)

    ax.contour(X, Y, pdf_values, colors='black')
    ax.grid(False, which='both')

    ax.set_xlabel("$X$ in nm")
    ax.set_ylabel("$X_{\\mathrm{b}}$ in nm")
    ax.set_title("Distribution of $X$ and $X_{\\mathrm{b}}$" +f" at t={t}")
    #ax.set_aspect('equal')
    fig.colorbar(heatmap_mesh, ax=ax, label="probability density in nm$^{-2}$")
    ensure_parent_dir(save_path) 
    plt.savefig(save_path)
    plt.close(fig)
    return fitted_distribution

def plot_heatmap_stationary(t_transient, timestep, trajectories, number_of_bins, save_path, random_number_generator, variable_indices=np.array([1,2])):
    """plot a heatmap of two random variables in stationary state, samples from multiple trajectories and times in stationary state, fit a gaussian to that distribution.

    Note: assumes first variable to be X second to be X_b in axis labels, rework for other use cases

    Parameters:
    t_transient: float: time at which the distribution is plotted
    timestep: float, time resolution of trajectories 
    number_of_bins: int, number of bins to divide sample space in
    save_path: str, path to save figure to
    random_number_generator: np.random rng, used to seed fitted distribution for comparability
    title: str, title of the plot
    variable_indices: array, optional: rows of trajectories of which histogram is to be plotted, default [1,2} (X,X_b)

    Returns:
    scipy.stats.multivariate_gaussian obtained by fit to data"""
    positions_stationary = format_trajectories_stationary(trajectories,t_transient,timestep,variable_indices)*POSITION_SCALE

    counts, xedges, yedges = np.histogram2d(
        positions_stationary[0], positions_stationary[1], bins=number_of_bins, density=True
    )
    fitted_distribution = fit_gaussian(format_trajectories_stationary(trajectories,t_transient, timestep, variable_indices),random_number_generator)
    fig, ax = create_fig()
    plot_dist = scale_gaussian(fitted_distribution, POSITION_SCALE)
    heatmap_mesh = ax.pcolormesh(xedges, yedges, counts.T, shading='auto')

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    grid_points = np.dstack((X, Y))
    pdf_values = plot_dist.pdf(grid_points)

    ax.contour(X, Y, pdf_values, colors='black')
    ax.grid(False, which='both')
    ax.set_xlabel("$X$ in nm")
    ax.set_ylabel("$X_{\\mathrm{b}}$ in nm")
    ax.set_title("Distribution of $X$ and $X_{\\mathrm{b}}$ in stationary state")
    #ax.set_aspect('equal')
    fig.colorbar(heatmap_mesh, ax=ax, label="probability density in nm$^{-2}$")
    ensure_parent_dir(save_path) 
    plt.savefig(save_path)
    plt.close(fig)
    return(fitted_distribution)

def plot_pdf_difference(x_centers, y_centers, fitted_pdf, analytical_pdf, save_path):
    """Create a heatmap, intended for use in compare-fitted_and_analytical-distribution

    Parameters: 
    x_centers_y_centers: arrays of the x/y coordinates of the points on which the distribution are to be compared
    fitted_pdf, analytical_pdf: arrays, of the two distribution evaluated on all points given by x,y_centers
    save_path: str, path to save figure to"""
    X, Y = np.meshgrid(x_centers, y_centers)
    diff = fitted_pdf - analytical_pdf

    fig, ax = create_fig()
    ax.grid(False, which='both')
    max_abs = np.max(np.abs(diff))
    mesh = ax.pcolormesh(X, Y, diff, cmap='seismic', vmin=-max_abs, vmax=max_abs, shading='auto')
    fig.colorbar(mesh, ax=ax, label="fitted $-$ analytical in nm$^{-2}$")
    ax.set_xlabel("$X$ in nm")
    ax.set_ylabel("$X_{\\mathrm{b}}$ in nm")
    ensure_parent_dir(save_path) 
    plt.savefig(save_path)
    plt.close(fig)

def compare_fitted_and_analytical(fitted_distribution, analytical_distribution, x_range, y_range, save_path, number_of_points=200):
    """Wraps plot_pdf_difference to plot to plot a heatmap of the differece of two 2d gaussians.
    
    parameters:
    fitted_distribution: scipy.stats.multivariate_normal, first distribution to compare
    analytical_distribution: scipy.stats.multivariate_normal, second distribution to compare
    x_range, y_range: arrays with two entries, a lower and an upper bound of the range on which the variables should be compared
    save_path: str, path to save figure to
    number_of-points, optional, default 200, number of points to discretize the ranges into"""

    fitted_display = scale_gaussian(fitted_distribution, POSITION_SCALE)
    analytical_display = scale_gaussian(analytical_distribution, POSITION_SCALE)
    x_centers = np.linspace(x_range[0], x_range[1], number_of_points)
    y_centers = np.linspace(y_range[0], y_range[1], number_of_points)
    X, Y = np.meshgrid(x_centers, y_centers)
    grid_points = np.dstack((X, Y))

    fitted_pdf = fitted_display.pdf(grid_points)#evaluate pdfs on all ponts in the discretized ranges
    analytical_pdf = analytical_display.pdf(grid_points)

    plot_pdf_difference(x_centers, y_centers, fitted_pdf, analytical_pdf, save_path)



def plot_autocorrelation(trajectories, t_final,resolution,time_step, t_transient, noise_amplitude, parameters, save_path):
    """plot the autocorrelation of an OUP in the stationary state from simulation and analytical considerations

    Parameters:
    trajectories: np.ndarray as returned by the integrators from the integrator module
    t_final: float, end of simulation
    t_transient: float: time at which the distribution is plotted
    time_step: float, time resolution of trajectories 
    noise_amplitude: np.array containing the prefactor of the random numbers for each entry of a current_state, i.e X,X_b
    parameters: SystemVariables
    save_path: str, path to save figure to
    """
    number_of_datapoints = round((t_final-t_transient)/resolution) +1
    results = np.empty((2,number_of_datapoints))#Use all position values that have the giiven toime difference -> less samples for larger differences
    for k in range(number_of_datapoints):
        time_difference= k*resolution
        results[0,k] = time_difference
        results[1,k] = stationary_correlation(trajectories, time_difference,time_step, t_transient)
    
    fig, ax = create_fig()
    ax.set_xlabel("Time difference in s")
    ax.set_ylabel("Autocorrelation in nm$^2$")
    ax.set_title("Autocorrelation in stationary state")
    ax.plot(results[0], results[1]*POSITION_SCALE**2, label = "Numerical")
    ax.plot(results[0], OUP_stationary_autocorrrelation(results[0], parameters)*POSITION_SCALE**2, label = "Analytical")
    ax.legend()
    ensure_parent_dir(save_path) 
    plt.savefig(save_path)
    plt.close(fig)

def plot_autocorrelation_to_init(trajectories, t_final,resolution,time_step, analytical_autocorrelation, parameters, save_path, variable_indices, title):
    """plot the autocorrelation at atime to the initial time of a random variable and compare with analytical solution. This is useful if we do not have a stationary state

    Parameters:
    trajectories: np.ndarray as returned by the integrators from the integrator module
    t_final: float, end of simulation
    t_transient: float: time at which the distribution is plotted
    time_step: float, time resolution of trajectories 
    resolution: float, larger then time step, time step for timedifferences for which autocorrelation is computed
    analytical_autocorrelation: function giving the analytical autocorrelation, signature time,parameters
    parameters: SystemVariables
    save_path: str, path to save figure to
    variable_indices: array containing the index on trajectories of the variable of which the autocorrelation is to be plotted
    title: str, title of the plot
    """
    number_of_datapoints = round(t_final/resolution) +1
    results = np.empty((2,number_of_datapoints))
    for k in range(0,number_of_datapoints):
        t= k*resolution
        results[0,k] = t
        results[1,k] = correlation_to_init_time(trajectories, t,time_step, variable_indices)
    fig, ax = create_fig()
    ax.set_xlabel("Time difference in s")
    ax.set_ylabel("Autocorrelation in nm$^2$")
    ax.set_title(title)
    ax.plot(results[0], results[1]*POSITION_SCALE**2, label = "Numerical")
    ax.plot(results[0], analytical_autocorrelation(results[0], parameters)*POSITION_SCALE**2, label = "Analytical")
    ax.legend()
    ensure_parent_dir(save_path) 
    plt.savefig(save_path)
    plt.close(fig)

def plot_trajectory(trajectories, trajectoy_number, variable_indices, save_path, labels):
    """plot time development of selected variables of a trajectory

    Parameters:
    trajectories: np.ndarray as returned by the integrators from the integrator module
    trajectory_number: int, trajectory to plot, has to be < then the number of simulated trajectories
    variable_indices: array of the indices of the variables to plot, for instance [1,2] for X, X_b
    save_path: str, path to save figure to
    labels: array of strings to label the plots of the individual trajectories"""

    fig, ax = create_fig()
    ax.set_xlabel("$t$ in s")
    ax.set_ylabel("Position in nm")
    ax.set_title("Example trajectory")
    for i in variable_indices:
        ax.plot(trajectories[trajectoy_number,0,:],trajectories[trajectoy_number,i,:]*POSITION_SCALE, label=labels[i-1])
    ax.legend()
    ensure_parent_dir(save_path) 
    plt.savefig(save_path)
    plt.close(fig)


    
def plot_two_normal_distributions(dist1, dist2, axis_labels, save_path, labels, 
                                    n_std=4,
                                    title="Comparison of Normal Distributions", 
                                    figsize=(10, 6)):
    """
        Plot two scipy.stats.norm distributions on the same graph.
        
        The x-axis range is automatically determined based on the mean and 
        standard deviation of both distributions, ensuring both curves are 
        fully visible.
        
        Parameters:

        dist1 : First normal distribution (scipy.stats.Normal)
        dist2 :  Second normal distribution 
        labels : tuple of str, optional
            Labels for (dist1, dist2)
        axis_labels: array of strings containing the names of x and y axis
        save-path: location to save plot
        n_std : float, optional
            Number of standard deviations to extend beyond each distribution's
            mean when calculating the plot range. Default is 4.
        title : str, optional
            Plot title.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        
    """
    dist1 = scale_gaussian(dist1, POSITION_SCALE)
    dist2 = scale_gaussian(dist2, POSITION_SCALE)
    n_points=1000
    # Extract parameters (mean/std) from each frozen distribution 
    mean1, std1 = dist1.mean[0], np.sqrt(dist1.cov[0,0])
    mean2, std2 = dist2.mean[0], np.sqrt(dist2.cov[0,0])
    
    # --- Automatically determine x-axis range ---
    x_min = min(mean1 - n_std * std1, mean2 - n_std * std2)
    x_max = max(mean1 + n_std * std1, mean2 + n_std * std2)
    x = np.linspace(x_min, x_max, n_points)
    
    # --- Compute PDFs ---
    pdf1 = dist1.pdf(x)
    pdf2 = dist2.pdf(x)
    
    
    # --- Plot ---figsize=figsize
    fig, ax = create_fig()
    
    ax.plot(x, pdf1, color="b", linewidth=2, label=labels[0])
    ax.plot(x, pdf2, color="r", linewidth=2, label=labels[1])
    
    
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.legend(loc="best")
    plt.close(fig)
    
    plt.tight_layout()
    ensure_parent_dir(save_path) 
    plt.savefig(save_path)

_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
_LINESTYLES = ['-', '--', '-.', ':']

def _style_for_index(index):
    """
    Return a (marker, linestyle) pair for the index-th series in a plot with
    multiple overlaid series distinguished primarily by color (e.g. one
    line per t_2 value). Color alone can fail to distinguish series whose
    curves nearly or exactly coincide. Markers and linestyles are cycled independently, so the
    combination only repeats after len(_MARKERS)*len(_LINESTYLES) series.

    Parameters:
    index : int
        Index of the series (e.g. position within a sorted t2_values list).

    Returns:
    marker, linestyle : str, str
    """
    marker = _MARKERS[index % len(_MARKERS)]
    linestyle = _LINESTYLES[(index // len(_MARKERS)) % len(_LINESTYLES)]
    return marker, linestyle

def try_overlay_matlab_protocol(ax, t_array, matlab_protocol_dir, t_second_measurement,
                                  t_protocol_end, m_0, m_t, color='black'):
    """
    Attempt to load a MATLAB-exported protocol for one (t2, tf, m_0, m_t)
    combination and overlay it as lambda(t) on an existing Axes, alongside
    a Python-computed optimal trajectory.

    Parameters:
    
    ax : matplotlib.axes.Axes
        Axes to draw onto; assumed to already show Python's X, X_b, lambda
        curves in nm (via POSITION_SCALE) against a time axis in seconds.
    t_array : np.ndarray
        Time values (seconds) at which to evaluate the MATLAB protocol;
        typically the same time grid as the Python optimal trajectory
        (`optimal_trajectories[0]`), so both curves are directly comparable.
        Any t outside the MATLAB export's own time range raises inside
        `load_matlab_protocol`'s interpolator (no extrapolation), so
        `t_array` should not exceed the MATLAB protocol's `t_protocol_end`.
    matlab_protocol_dir : str
        Directory containing MATLAB-exported protocol CSVs (as produced by
        `make_matlab_protocol_path`/`write_two_column_csv` on the MATLAB
        side), e.g. one of the `2state`/`3state` subfolders under
        `matlab_protocols/`.
    t_second_measurement, t_protocol_end : float
    m_0, m_t : int
    color : str, optional
        Line color for the overlaid MATLAB curve. Default 'black'.

    Returns
    -------
    bool
        True if a MATLAB protocol was found and plotted, False otherwise.
    """
    protocol_path = make_matlab_protocol_path(
        matlab_protocol_dir, t_second_measurement, t_protocol_end, m_0, m_t
    )
    try:
        matlab_lambda_func = load_matlab_protocol(protocol_path)
    except FileNotFoundError:
        return False

    lambda_matlab = np.array([matlab_lambda_func(t) for t in t_array])
    ax.plot(t_array, lambda_matlab * POSITION_SCALE, color=color,
            linestyle=':', label="$\\lambda_{\\mathrm{MATLAB}}$")
    return True

def _plot_lambda_jump(ax, lambda0, color):
    """
    Mark the trap's initial jump at t=0: a dotted vertical segment from 0
    (trap held at rest before this decision) up/down to lambda0 (the real
    first commanded position - compute_optimal_trajectories_array no longer
    forces this to 0, so a genuine jump is visible in the data, not just
    implicit as the curve's unremarkable starting point), plus a small open
    marker at the pre-jump rest reference. No-op when there's no jump (e.g.
    the (0,0) no-op state).

    Parameters:
    ax : matplotlib Axes
    lambda0 : float, lambda(t=0) already scaled to plotting units (nm)
    color : the lambda line's own color, so the jump reads as part of it
    """
    if lambda0 == 0:
        return
    ax.plot([0, 0], [0, lambda0], color=color, linestyle=":", lw=1.5)
    ax.plot(0, 0, marker="o", ms=4, mfc="white", mec=color)


def plot_optimal_trajectories(optimal_trajectories, title, save_path,
                               matlab_protocol_dir=None, t_second_measurement=None,
                               t_protocol_end=None, m_0=None, m_t=None):
    """create plot of optimal X,X_b and lambda. If matlab_protocol_dir is not
    None, the function checks for an existing matlab protocol and plots it as well

    Parameters:
    optimal_trajectories: ndarray containing rows for, time, X, X_b, lambda in that order
    title: str, title of the plot
    save_path: str, path to save location for the plot
    matlab_protocol_dir: str, where to check for matlab protocol
    t_second_measurement, t_protocol_end : float
    m_0, m_t : int"""

    fig, ax = create_fig()
    ax.set_xlabel("$t$ in s")
    ax.set_ylabel("Position in nm")
    ax.set_title(title)
    ax.plot(optimal_trajectories[0],optimal_trajectories[1]*POSITION_SCALE, label = "$x$")
    ax.plot(optimal_trajectories[0],optimal_trajectories[2]*POSITION_SCALE, label = "$x_{\\mathrm{b}}$")
    lambda_line, = ax.plot(optimal_trajectories[0],optimal_trajectories[3]*POSITION_SCALE, label = "$\\lambda$")
    _plot_lambda_jump(ax, optimal_trajectories[3, 0] * POSITION_SCALE, lambda_line.get_color())
    if matlab_protocol_dir is not None:
        try_overlay_matlab_protocol(
            ax, optimal_trajectories[0], matlab_protocol_dir,
            t_second_measurement, t_protocol_end, m_0, m_t,
        )
    ax.legend()
    ensure_parent_dir(save_path) 
    plt.savefig(save_path)   
    plt.close() 
    

def plot_optimal_trajectories_grid(t_second_measurement, t_protocol_end, save_dir, save_path,
                                    n_outcomes=2, matlab_protocol_dir=None):
    """
    Plot optimal X, X_b, lambda trajectories and lambda_matlab if matlab protocol is not None and a fil for the measurements and times exist
    for every measurement outcome
    combination (m_0, m_t) at a fixed (t_second_measurement, t_protocol_end),
    one subplot per combination, sharing a single legend.

    Requires process_combination to have already been run and saved
    results for this (t_second_measurement, t_protocol_end) for every
    combination in save_dir; combinations
    with no saved file (e.g. negligible-probability outcomes that were
    skipped) are shown with an empty, labeled subplot instead of failing.

    Parameters:
    
    t_second_measurement, t_protocol_end : float
    save_dir : str
        Directory containing the saved .npz optimal-protocol files.
    save_path : str
        Output image path.
    n_outcomes : int, optional
        Number of measurement outcome categories; default 2 
    matlab_protocol_dir: str, where to check for matlab protocol
    """
    labels = outcome_labels(n_outcomes)
    combinations = [(m_0, m_t) for m_0 in labels for m_t in labels]

    ncols = int(np.ceil(np.sqrt(len(combinations))))
    nrows = int(np.ceil(len(combinations) / ncols))

    fig_width = 6.5 * ncols
    fig_height = 5 * nrows
    fig = plt.figure(figsize=(fig_width, fig_height))

    line_colors = {"X": "tab:blue", "Xb": "tab:orange", "lambda": "tab:green"}
    matlab_found_anywhere = False

    for index, (m_0, m_t) in enumerate(combinations):
        ax = fig.add_subplot(nrows, ncols, index + 1)
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3), useMathText=True)
        ax.set_xlabel("$t$ in s")
        ax.set_ylabel("Position in nm")
        ax.set_title(f"$m_0={int(m_0)}$, $m_t={int(m_t)}$")

        try:
            data = load_optimal_protocol(save_dir, t_second_measurement, t_protocol_end, m_0, m_t)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax.transAxes)
            continue

        t, X, Xb, lam = data["optimal_trajectories"]
        ax.plot(t, X * POSITION_SCALE, color=line_colors["X"])
        ax.plot(t, Xb * POSITION_SCALE, color=line_colors["Xb"])
        ax.plot(t, lam * POSITION_SCALE, color=line_colors["lambda"])
        _plot_lambda_jump(ax, lam[0] * POSITION_SCALE, line_colors["lambda"])

        if matlab_protocol_dir is not None:
            found = try_overlay_matlab_protocol(
                ax, t, matlab_protocol_dir, t_second_measurement, t_protocol_end, m_0, m_t,
                color="black",
            )
            matlab_found_anywhere = matlab_found_anywhere or found

        handles = [
        Line2D([0], [0], color=line_colors["X"], label="$x$"),
        Line2D([0], [0], color=line_colors["Xb"], label="$x_{\\mathrm{b}}$"),
        Line2D([0], [0], color=line_colors["lambda"], label="$\\lambda$"),
        Line2D([0], [0], color=line_colors["lambda"], linestyle=":", label="$\\lambda$ jump at $t=0$"),
    ]

    if matlab_found_anywhere:
        handles.append(Line2D([0], [0], color="black", linestyle=":",
                               label="$\\lambda_{\\mathrm{MATLAB}}$"))
    add_suptitle_and_legend(
        fig, f"Optimal trajectories, $\\Delta t={t_second_measurement}$, $t_{{\\mathrm{{f}}}}={t_protocol_end}$",
        handles,
    )
    ensure_parent_dir(save_path) 
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_work_comparison(all_results, params, save_path):
    """
    Visualize mean work of the optimal protocol as a function of protocol
    duration t_f, faceted by measurement outcome (m_0, m_t) and colored by
    time of second measurement t_2 (Delta t), with a dashed horizontal
    reference line per t_2 showing -<V_trap> at the true second moment
    E[X_t^2 | m_0, m_t]

    Parameters
    ----------
    all_results : dict
        Mapping (t_second_measurement, t_protocol_end, m_0, m_t) -> dict
        with keys "optimal", "optimal_check", "jump", "-V_trap", as
        produced by `process_combination`.
    params : SystemVariables
        Used only for k_B*T, to express work in dimensionless k_BT units.
    save_path : str
        Output image path.

    Returns
    -------
    None
    """
    kT = params.k_B * params.T

    outcomes = sorted(set((m_0, m_t) for (_, _, m_0, m_t) in all_results.keys()))
    t2_values = sorted(set(t2 for (t2, _, _, _) in all_results.keys()))

    n_outcomes = len(outcomes)
    ncols = int(np.ceil(np.sqrt(n_outcomes)))
    nrows = int(np.ceil(n_outcomes / ncols))

    fig_width = 6.5 * ncols
    fig_height = 5 * nrows
    fig = plt.figure(figsize=(fig_width, fig_height))
    colors = plt.cm.viridis(np.linspace(0, 1, len(t2_values)))

    axes = []
    for index, (m_0, m_t) in enumerate(outcomes):
        ax = fig.add_subplot(nrows, ncols, index + 1)
        axes.append(ax)
        # Reproduce create_fig()'s per-axes formatting (create_fig() itself
        # only builds a single fig/ax pair, so its formatting is applied
        # here manually to each grid cell instead).
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3), useMathText=True)

        for index, (color, t2) in enumerate(zip(colors, t2_values)):
            marker, linestyle = _style_for_index(index)
            points = sorted(
                (tf, work["optimal"] / kT, work["-V_trap"] / kT)
                for (t2_, tf, m0_, mt_), work in all_results.items()
                if t2_ == t2 and m0_ == m_0 and mt_ == m_t
            )
            if not points:
                continue
            tf_arr, work_arr, ref_arr = zip(*points)

            ax.plot(tf_arr, work_arr, marker=marker, linestyle=linestyle, color=color)
            ax.axhline(ref_arr[0], color=color, linestyle='--', alpha=0.6)

        ax.set_xlabel("$t_{\\mathrm{f}}$ in s")
        ax.set_ylabel("$W/k_BT$")
        ax.set_title(f"$m_0={int(m_0)}$, $m_t={int(m_t)}$")

    color_handles = [
        Line2D([0], [0], color=color, marker='o', label=f"$\\Delta t={t2}$")
        for color, t2 in zip(colors, t2_values)
    ]
    reference_handle = Line2D([0], [0], color='gray', linestyle='--',
                               label=r"$-\langle V_{\mathrm{trap}}\rangle_{m_0,m_t}$ (reference)")

    add_suptitle_and_legend(
        fig, "Mean work of the optimal protocol vs. protocol duration",
        color_handles + [reference_handle],
    )
    ensure_parent_dir(save_path) 
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_work_gap(all_results, params, save_path):
    """
    Visualize difference of mean work of the optimal protocol and -<V_trap> evaluated at the true second moment
    E[X_t^2 | m_0, m_t], as a function of protocol
    duration t_f, faceted by measurement outcome (m_0, m_t) and colored by
    time of second measurement t_2 (Delta t), with a dashed horizontal
    reference line per t_2 showing -<V_trap>, the (t_f-independent)
    negative mean potential energy 


    Parameters
    ----------
    all_results : dict
        Mapping (t_second_measurement, t_protocol_end, m_0, m_t) -> dict
        with keys "optimal", "optimal_check", "jump", "-V_trap", as
        produced by `process_combination`.
    params : SystemVariables
        Used only for k_B*T, to express work in dimensionless k_BT units.
    save_path : str
        Output image path.

    Returns
    -------
    None
    """
    kT = params.k_B * params.T

    outcomes = sorted(set((m_0, m_t) for (_, _, m_0, m_t) in all_results.keys()))
    t2_values = sorted(set(t2 for (t2, _, _, _) in all_results.keys()))

    n_outcomes = len(outcomes)
    ncols = int(np.ceil(np.sqrt(n_outcomes)))
    nrows = int(np.ceil(n_outcomes / ncols))

    fig_width = 6.5 * ncols
    fig_height = 5 * nrows
    fig = plt.figure(figsize=(fig_width, fig_height))
    colors = plt.cm.viridis(np.linspace(0, 1, len(t2_values)))

    axes = []
    for index, (m_0, m_t) in enumerate(outcomes):
        ax = fig.add_subplot(nrows, ncols, index + 1)
        axes.append(ax)
        # Reproduce create_fig()'s per-axes formatting (create_fig() itself
        # only builds a single fig/ax pair, so its formatting is applied
        # here manually to each grid cell instead).
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3), useMathText=True)

        for index, (color, t2) in enumerate(zip(colors, t2_values)):
            marker, linestyle = _style_for_index(index)
            points = sorted(
                (tf, (work["optimal"]/work["-V_trap"]))
                for (t2_, tf, m0_, mt_), work in all_results.items()
                if t2_ == t2 and m0_ == m_0 and mt_ == m_t
            )
            if not points:
                continue
            tf_arr, difference_arr = zip(*points)

            ax.plot(tf_arr, difference_arr, marker=marker, linestyle=linestyle, color=color)

        ax.axhline(1, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("$t_{\\mathrm{f}}$ in s")
        ax.set_ylabel("$W/\\left(-\\langle V_{\\mathrm{trap}}\\rangle_{m_0,m_t}\\right)$")
        ax.set_title(f"$m_0={int(m_0)}$, $m_t={int(m_t)}$")

    color_handles = [
        Line2D([0], [0], color=color, marker='o', label=f"$\\Delta t={t2}$")
        for color, t2 in zip(colors, t2_values)
    ]

    add_suptitle_and_legend(
        fig, "Ratio of optimal-protocol work and $-\\langle V_{\\mathrm{trap}}\\rangle_{m_0,m_t}$ vs. protocol duration",
        color_handles,
    )
    ensure_parent_dir(save_path) 
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_initial_condition_deviation_grid(t_second_measurement, t_protocol_end, trajectories, save_dir, save_path,
                                    n_outcomes=2,):
    labels = outcome_labels(n_outcomes)
    combinations = [(m_0, m_t) for m_0 in labels for m_t in labels]

    ncols = int(np.ceil(np.sqrt(len(combinations))))
    nrows = int(np.ceil(len(combinations) / ncols))

    fig_width = 6.5 * ncols
    fig_height = 5 * nrows
    fig = plt.figure(figsize=(fig_width, fig_height))
    

    for index, (m_0, m_t) in enumerate(combinations):
        ax = fig.add_subplot(nrows, ncols, index + 1)
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3), useMathText=True)
        ax.set_xlabel("Number of trajectories")
        ax.set_ylabel("Position in nm")
        ax.set_title(f"$m_0={int(m_0)}$, $m_t={int(m_t)}$")

        try:
            data = load_optimal_protocol(save_dir, t_second_measurement, t_protocol_end, m_0, m_t)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax.transAxes)
            continue

        t, X, Xb, lam = data["optimal_trajectories"]


        handles = [
        Line2D([0], [0], color=line_colors["X"], label="$X$"),
        Line2D([0], [0], color=line_colors["Xb"], label="$X_{\\mathrm{b}}$"),
        Line2D([0], [0], color=line_colors["lambda"], label="$\\lambda$"),
    ]

    add_suptitle_and_legend(
        fig, f"Optimal trajectories, $\\Delta t={t_second_measurement}$, $t_{{\\mathrm{{f}}}}={t_protocol_end}$",
        handles,
    )
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_accumulated_work(t_second_measurement, t_protocol_end, save_dir, save_path, params, n_outcomes=2):
    """plot the accumulated work against time to visualize when most work is done
    
    Parameters:
    t_second_measurement, t_protocol_end : float
    save_dir : str
        Directory containing the saved .npz optimal-protocol files.
    save_path : str
        Output image path.
    n_outcomes : int, optional
        Number of measurement outcome categories; default 2 
    params: SystemVariables
    """
    labels = outcome_labels(n_outcomes)
    combinations = [(m_0, m_t) for m_0 in labels for m_t in labels]

    ncols = int(np.ceil(np.sqrt(len(combinations))))
    nrows = int(np.ceil(len(combinations) / ncols))

    fig_width = 6.5 * ncols
    fig_height = 5 * nrows
    fig = plt.figure(figsize=(fig_width, fig_height))
    

    for index, (m_0, m_t) in enumerate(combinations):
        ax = fig.add_subplot(nrows, ncols, index + 1)
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3), useMathText=True)
        ax.set_xlabel("$t$ in s")
        ax.set_ylabel("$\\langle W \\rangle$ in $k_{\\mathrm{B}}T$")
        ax.set_title(f"$m_0={int(m_0)}$, $m_t={int(m_t)}$")

        try:
            data = load_optimal_protocol(save_dir, t_second_measurement, t_protocol_end, m_0, m_t)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax.transAxes)
            continue
        
        k_BT =params.k_B *params.T
        work_time_array = compute_work_per_step(params,data["optimal_trajectories"])
        total_work = data["mean_work"]/k_BT

        ax.plot(work_time_array[0], work_time_array[2]/k_BT, color="tab:blue")
        ax.axhline(total_work, color='grey', linestyle='--', alpha=0.5)

    add_suptitle_and_legend(fig, f"Accumulated work, $\\Delta t={t_second_measurement}$, $t_{{\\mathrm{{f}}}}={t_protocol_end}$")
    ensure_parent_dir(save_path) 
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
        

def plot_bootstrap_convergence_grid(trajectories, t_second_measurement, time_step, params,
                                     analytical_positions, full_sizes,
                                     save_path_particle, save_path_bath,random_number_generator,
                                     subsample_fraction=0.3, n_bootstrap=200,
                                     n_outcomes=2,
                                     ):
    """
    For every measurement outcome (m_0, m_t), compute means for multiple "full pool" sizes, and produce separate
    faceted grid figures: one for X and one for X_b

    Parameters:
    trajectories : np.ndarray
        Full array of simulated trajectories, as returned by
        `euler_maruyama_ensemble_bath_model`.
    t_second_measurement : float
    time_step : float
        Output time step at which `trajectories` was saved.
    params : SystemVariables
    analytical_positions : dict
        As returned by `mean_initial_positions_from_analytical`: keyed by
        "m_0=...,m_t=..."
    full_sizes : sequence of int
        The "full pool" sizes to run a separate bootstrap study at 
    save_path_particle : str
        Output path for the X grid.
    save_path_bath : str
        Output path for the X_b grid.
    subsample_fraction, n_bootstrap, with_replacement, random_number_generator :
        Passed through to `bootstrap_initial_positions_from_trajectories`
        for every full_size.
    n_outcomes : int, optional
        Default 2.
    """
    random_number_generator = random_number_generator if random_number_generator is not None else np.random.default_rng()
    sorted_full_sizes = sorted(full_sizes)

    # One bootstrap study per full_size, each internally covering all
    # outcome combinations and both X and X_b
    bootstrap_results_by_full_size = {
        full_size: bootstrap_initial_positions_from_trajectories(
            trajectories, t_second_measurement, time_step, params,
            full_size=full_size, subsample_fraction=subsample_fraction,
            n_bootstrap=n_bootstrap, n_outcomes=n_outcomes, random_number_generator=random_number_generator,
        )
        for full_size in sorted_full_sizes
    }

    labels = outcome_labels(n_outcomes)
    combinations = [(m_0, m_t) for m_0 in labels for m_t in labels]

    def draw_convergence_grid(quantity_key, standard_deviation_key, axis_label, plot_title, save_path):
        """Draw one faceted convergence grid (all outcome combinations) for a single quantity."""
        number_of_columns = int(np.ceil(np.sqrt(len(combinations))))
        number_of_rows = int(np.ceil(len(combinations) / number_of_columns))
        figure = plt.figure(figsize=(6.5 * number_of_columns, 5 * number_of_rows))

        for index, (m_0, m_t) in enumerate(combinations):
            axes = figure.add_subplot(number_of_rows, number_of_columns, index + 1)
            axes.grid(which='major', color='#CCCCCC', linestyle='--')
            axes.grid(which='minor', color='#CCCCCC', linestyle=':')
            axes.xaxis.set_minor_locator(AutoMinorLocator(5))
            axes.yaxis.set_minor_locator(AutoMinorLocator(5))
            axes.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3), useMathText=True)
            axes.set_xscale('log')
            axes.set_xlabel("Number of trajectories")
            axes.set_ylabel(axis_label)
            axes.set_title(f"$m_0={int(m_0)}$, $m_t={int(m_t)}$")

            outcome_key = f"m_0={m_0},m_t={m_t}"
            if outcome_key not in analytical_positions or np.isnan(analytical_positions[outcome_key][quantity_key]):
                axes.text(0.5, 0.5, "no data", ha='center', va='center', transform=axes.transAxes)
                continue

            bootstrap_means = np.array([
                bootstrap_results_by_full_size[full_size][outcome_key][quantity_key]
                for full_size in sorted_full_sizes
            ])
            bootstrap_standard_deviations = np.array([
                bootstrap_results_by_full_size[full_size][outcome_key][standard_deviation_key]
                for full_size in sorted_full_sizes
            ])
            analytical_value = analytical_positions[outcome_key][quantity_key]

            axes.errorbar(sorted_full_sizes, bootstrap_means * POSITION_SCALE,
                           yerr=bootstrap_standard_deviations * POSITION_SCALE,
                           fmt='o', color='tab:blue', capsize=4)
            axes.axhline(analytical_value * POSITION_SCALE, color='black', linestyle='--')

        legend_handles = [
            Line2D([0], [0], marker='o', linestyle='', color='tab:blue', label="Bootstrap estimate from trajectories"),
            Line2D([0], [0], color='black', linestyle='--', label="Analytical"),
        ]
        add_suptitle_and_legend(figure, plot_title, legend_handles)
        ensure_parent_dir(save_path) 
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(figure)

    draw_convergence_grid(
        "mean_X_t", "std_X_t", "Mean $X$ in nm",
        f"Bootstrap convergence of $X$, $\\Delta t={t_second_measurement}$",
        save_path_particle,
    )
    draw_convergence_grid(
        "mean_X_bt", "std_X_bt", "Mean $X_{\\mathrm{b}}$ in nm",
        f"Bootstrap convergence of $X_{{\\mathrm{{b}}}}$, $\\Delta t={t_second_measurement}$",
        save_path_bath,
    )

def plot_stochastic_trajectories_grid(t_second_measurement, t_protocol_end, save_dir, save_path, params,
                                        random_number_generator, number_of_samples=100,
                                        n_trajectories_to_plot=50, n_outcomes=2,
                                        fine_time_step=FINE_TIME_STEP_NON_EQ, target_number_of_saves=500):
    """
    For every measurement outcome combination (m_0, m_t) at a fixed
    (t_second_measurement, t_protocol_end), simulate a stochastic ensemble
    under the saved optimal protocol and plot a subset of trajectories
    together with the deterministic optimum, one subplot per combination.

    Requires process_combination to have already saved a protocol file for
    this (t_second_measurement, t_protocol_end) under save_dir for every
    combination

    Parameters:
    t_second_measurement, t_protocol_end : float
    save_dir : str
        Directory containing the saved .npz optimal-protocol files.
    save_path : str
    params : SystemVariables
    random_number_generator : numpy.random.Generator
    number_of_samples : int, optional
        Number of stochastic replicates simulated per outcome. Default 200.
    n_trajectories_to_plot : int, optional
        Number of individual replicates drawn as thin lines per subplot.
        Default 50.
    n_outcomes : int, optional
        Default 2.
    fine_time_step : float, optional
        Integration step; defaults to FINE_TIME_STEP_NON_EQ 
    target_number_of_saves : int, optional
        Passed to simulate_stochastic_trajectories_under_optimal_protocol.
        Default 1000.
    """
    labels = outcome_labels(n_outcomes)
    combinations = [(m_0, m_t) for m_0 in labels for m_t in labels]

    ncols = int(np.ceil(np.sqrt(len(combinations))))
    nrows = int(np.ceil(len(combinations) / ncols))
    fig = plt.figure(figsize=(6.5 * ncols, 5 * nrows))

    for index, (m_0, m_t) in enumerate(combinations):
        ax = fig.add_subplot(nrows, ncols, index + 1)
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3), useMathText=True)
        ax.set_xlabel("$t$ in s")
        ax.set_ylabel("Position in nm")
        ax.set_title(f"$m_0={int(m_0)}$, $m_t={int(m_t)}$")

        try:
            data = load_optimal_protocol(save_dir, t_second_measurement, t_protocol_end, m_0, m_t)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax.transAxes)
            continue

        stochastic_trajectories = simulate_stochastic_trajectories_under_optimal_protocol(
            data["c_vector"], data["lambda_f"], t_protocol_end, params,
            number_of_samples, fine_time_step, random_number_generator,
            target_number_of_saves=target_number_of_saves,
        )

        t_stochastic = stochastic_trajectories[0, 0, :]
        X_all = stochastic_trajectories[:, 1, :]
        Xb_all = stochastic_trajectories[:, 2, :]
        n_show = min(n_trajectories_to_plot, X_all.shape[0])

        for i in range(n_show):
            ax.plot(t_stochastic, X_all[i] * POSITION_SCALE, color="tab:blue", alpha=0.15, linewidth=0.8)
            ax.plot(t_stochastic, Xb_all[i] * POSITION_SCALE, color="tab:orange", alpha=0.15, linewidth=0.8)

        ax.plot(t_stochastic, X_all.mean(axis=0) * POSITION_SCALE, color="tab:blue", linewidth=2.2)
        ax.plot(t_stochastic, Xb_all.mean(axis=0) * POSITION_SCALE, color="tab:orange", linewidth=2.2)

        opt = data["optimal_trajectories"]
        ax.plot(opt[0], opt[1] * POSITION_SCALE, color="black", linestyle="--", linewidth=1.8)
        ax.plot(opt[0], opt[2] * POSITION_SCALE, color="dimgray", linestyle="--", linewidth=1.8)
        ax.plot(opt[0], opt[3] * POSITION_SCALE, color="tab:green", linewidth=1.8)

    handles = [
        Line2D([0], [0], color="tab:blue", label="Simulated mean $X$"),
        Line2D([0], [0], color="tab:orange", label="Simulated mean $X_{\\mathrm{b}}$"),
        Line2D([0], [0], color="black", linestyle="--", label="Optimal $X$"),
        Line2D([0], [0], color="dimgray", linestyle="--", label="Optimal $X_{\\mathrm{b}}$"),
        Line2D([0], [0], color="tab:green", label="$\\lambda(t)$"),
    ]
    add_suptitle_and_legend(
        fig, f"Stochastic trajectories around optimal protocol, $\\Delta t={t_second_measurement}$, $t_{{\\mathrm{{f}}}}={t_protocol_end}$",
        handles,
    )
    ensure_parent_dir(save_path)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


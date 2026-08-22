"""
Compare MATLAB's ML-trained protocol (conditioned on Python's analytical
initial conditions) against Python's own closed-form optimal protocol -
both evaluated using Python's compute_work_general_protocol, so any
discrepancy reflects only how close MATLAB's learned protocol gets to the
true optimum, not any difference in work-evaluation convention.
"""
import os
import time
import numpy as np

from sdesim.params import params_bath_model
from sdesim.helpers import (
    load_matlab_protocol, make_matlab_protocol_path,
    parse_m_values, outcome_labels,
)
from sdesim.analytics import (
    mean_initial_positions_from_analytical, SolutionMatrixCreator, compute_D,
    get_cvector, mean_work, compute_work_general_protocol, current_lambda,
)

MATLAB_PROTOCOL_BASE_DIR = "matlab_protocols"

# Maps n_outcomes -> the subfolder MATLAB exports for that outcome
# convention are stored in (as seen in MATLAB_PROTOCOL_BASE_DIR).
MATLAB_STATE_SUBDIR = {
    2: "2state",
    3: "3state",
}


def matlab_protocol_dir(n_outcomes):
    """
    Return the directory containing MATLAB-exported protocols for a given
    number of measurement outcomes.

    Parameters
    ----------
    n_outcomes : int
        Number of measurement outcome categories (e.g. 2 or 3); must be a
        key of `MATLAB_STATE_SUBDIR`.

    Returns
    -------
    str
        Path `MATLAB_PROTOCOL_BASE_DIR/<subdir>`, matching the on-disk
        layout produced by the MATLAB export side.
    """
    if n_outcomes not in MATLAB_STATE_SUBDIR:
        raise ValueError(f"No MATLAB export subdirectory known for n_outcomes={n_outcomes}; "
                          f"add it to MATLAB_STATE_SUBDIR if this convention exists.")
    return os.path.join(MATLAB_PROTOCOL_BASE_DIR, MATLAB_STATE_SUBDIR[n_outcomes])


def compare_one(t_second_measurement, t_protocol_end, m_0, m_t, params, n_outcomes=2):
    """
    Evaluate MATLAB's learned protocol and Python's closed-form optimal
    protocol for one (t2, tf, m_0, m_t) combination, both using the
    identical Python initial condition and the identical Python work
    evaluator (compute_work_general_protocol), so any discrepancy reflects
    only how close MATLAB's learned protocol gets to the true optimum .

    Parameters
    ----------
    t_second_measurement, t_protocol_end : float
    m_0, m_t : int
    params : SystemVariables
    n_outcomes : int, optional
        Default 2.

    Returns
    -------
    dict or None
        None if this outcome has negligible probability under the model
        (mean_X_t is NaN) or no matching MATLAB export file is found.
    """
    positions = mean_initial_positions_from_analytical(params, t_second_measurement, n_outcomes)
    key = f"m_0={m_0},m_t={m_t}"
    mean_X_t = positions[key]["mean_X_t"]
    mean_X_bt = positions[key]["mean_X_bt"]

    if np.isnan(mean_X_t):
        return None

    protocol_path = make_matlab_protocol_path(
        matlab_protocol_dir(n_outcomes), t_second_measurement, t_protocol_end, m_0, m_t
    )
    try:
        matlab_lambda_func = load_matlab_protocol(protocol_path)
    except FileNotFoundError:
        return None

    work_matlab = compute_work_general_protocol(
        mean_X_t, mean_X_bt, matlab_lambda_func, t_protocol_end, params
    )

    B_creator = SolutionMatrixCreator(params)
    D = compute_D(B_creator, t_protocol_end, params)
    c_vector, lambda_f = get_cvector(mean_X_t, mean_X_bt, D)

    def lambda_optimal(t, c_vector=c_vector):
        B = B_creator.current_matrix(t)
        return current_lambda(B, c_vector, params)

    work_python_optimal = compute_work_general_protocol(
        mean_X_t, mean_X_bt, lambda_optimal, t_protocol_end, params, lambda_f=lambda_f
    )

    return {
        "work_matlab": work_matlab,
        "work_python_optimal": work_python_optimal,
        "gap": work_matlab - work_python_optimal,
    }


def main():
    # Each entry: (t2, tf, n_outcomes) - n_outcomes determines both the
    # outcome_labels() convention and which MATLAB subfolder is read.
    t2_tf_pairs = [(1, 3, 2), (1, 5, 3)]

    kT = params_bath_model.k_B * params_bath_model.T
    print("Comparison of MATLAB (ML-trained) vs. Python (closed-form optimal) protocol work:\n")

    for t2, tf, n_outcomes in t2_tf_pairs:
        labels = outcome_labels(n_outcomes)
        for m_0 in labels:
            for m_t in labels:
                result = compare_one(t2, tf, m_0, m_t, params_bath_model, n_outcomes)
                if result is None:
                    continue
                print(f"n_outcomes={n_outcomes}, t2={t2}, tf={tf}, m_0={m_0}, m_t={m_t}:")
                print(f"  MATLAB (ML)      = {result['work_matlab']/kT:.4f} kBT")
                print(f"  Python (optimal) = {result['work_python_optimal']/kT:.4f} kBT")
                print(f"  gap "
                      f"= {result['gap']/kT:.4f} kBT\n")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
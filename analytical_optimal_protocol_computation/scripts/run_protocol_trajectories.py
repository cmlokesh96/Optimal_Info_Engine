"""
Simulate stochastic ensembles under previously computed optimal protocols
and visualize them for every measurement outcome, across multiple (t_second_measurement,
t_protocol_end) combinations.

Requires process_combination to have already been run and saved protocol
files under SAVE_DIR_OPTIMAL_PROTOCOLS for these time combinations (e.g.
via run_optimal_protocol.py).
"""
import time
import numpy as np

from sdesim.constants import SAVE_DIR_OPTIMAL_PROTOCOLS, PLOT_DIR_OPTIMAL_PROTOCOLS, SEED_5
from sdesim.params import params_bath_model
from sdesim.visualization import setup_matplotlib, plot_stochastic_trajectories_grid


def main():
    setup_matplotlib()
    rng = np.random.default_rng(SEED_5)

    #The times have to match protocols already saved by run_optimal_protocol.py
    t_second_measurement_values = [0, 0.5, 1, 3, 80]
    t_protocol_end_values = [1, 2, 2.5, 3, 3.5, 4, 5]

    for t_second_measurement in t_second_measurement_values:
        for t_protocol_end in t_protocol_end_values:
            save_path = (
                f"{PLOT_DIR_OPTIMAL_PROTOCOLS}/stochastic_validation_grid_"
                f"t2_{t_second_measurement}_tf_{t_protocol_end}.png"
            )
            plot_stochastic_trajectories_grid(
                t_second_measurement, t_protocol_end,
                save_dir=SAVE_DIR_OPTIMAL_PROTOCOLS,
                save_path=save_path,
                params=params_bath_model,
                random_number_generator=rng,
                number_of_samples=200,
                n_trajectories_to_plot=50,
                n_outcomes=2,
            )
            print(f"Saved {save_path}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
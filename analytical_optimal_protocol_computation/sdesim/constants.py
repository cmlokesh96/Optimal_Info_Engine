""""This file contains variables that should not be changed in any of the programs, indicated by the ALL CAPS convention, for instance file paths"""
import numpy as np 

#The mother seed is an 128 Bit integer as recomended by numpy for seeding. It was generated using secrets. 
MOTHER_SEED = 71798102969261776543995128308894035365

#SeedSequence is a method to create 3 independent seeds from one mother seed, one for each program
seed_sequence = np.random.SeedSequence(MOTHER_SEED)

SEED_1, SEED_2, SEED_3, SEED_4, SEED_5 = seed_sequence.spawn(5)

OUP_TRAJETORY_PATH = "plots/example_trajectory_OUP"
OUP_AUTOCORRELATION_PATH = "plots/autocorrelation_OUP"
OUP_HISTOGRAM_PATH = "plots/X_histogram_OUP"

BATH_EQ_TRAJECTORIES_PATH = "plots/example_trajectories_X_Xb"
BATH_EQ_HEATMAP_PATH = "plots/heatmap_x_xb"
BATH_EQ_DIFFERENCE_HEATMAP_PATH = "plots/analytical_fitted_difference_heatmap"
BATH_EQ_FITTED_2D_DISTRIBUTION_PATH = "parameters/fitted2D_Gaussian.npz"

BATH_NEQ_HEATMAP_PATH = "plots/heatmap_x_xb_non_eq"
BATH_NEQ_AUTOCORRELATION_X_0_X_t_PATH = "plots/autocorrelation_x_0_x_t_neq"
BATH_NEQ_AUTOCORRELATION_X_0_X_bt_PATH = "plots/autocorrelation_x_0_x_bt_neq"
BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH = "parameters/fitted4D_Gaussian.npz"

COMPARISON_P_X_T_X_0_PATH = "plots/P_X_T_X_0_comparison"
COMPARISON_P_X_b_T_X_0_PATH = "plots/P_X_b_T_X_0_comparison"

NON_EQUILIBRIUM_TRAJECTORIES_PATH = "parameters/non_eq_trajectories.npy"
TIME_STEP_NON_EQ = 0.5
FINE_TIME_STEP_NON_EQ = 0.001
OPTIMAL_PROTOCOL_PATHS = ["plots/optimal_trajectories_m_0_0_m_t_0", "plots/optimal_trajectories_m_0_0_m_t_1", "plots/optimal_trajectories_m_0_1_m_t_0", "plots/optimal_trajectories_m_0_1_m_t_1"]

SAVE_DIR_OPTIMAL_PROTOCOLS = "parameters/optimal_protocols"
PLOT_DIR_OPTIMAL_PROTOCOLS = "plots/optimal_protocols"

SAVE_DIR_OPTIMAL_PROTOCOLS_MULTIPLE_MEASUREMENT = "parameters/optimal_protocols_multiple_measurement"
PLOT_DIR_OPTIMAL_PROTOCOLS_MULTIPLE_MEASUREMENT = "plots/optimal_protocols_multiple_measurement"

SAVE_DIR_2_STATE_INIT = "parameters/2state_init_conditions"
SAVE_DIR_3_STATE_INIT = "parameters/3state_init_conditions"

PLOT_DIR_INITIAL_CONDITION_CONVERGENCE = "plots/initial_condition_convergence"


POSITION_SCALE = 1e9
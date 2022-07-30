from partxv2.partxInterface import run_partx
import numpy as np
from partxv2.bayesianOptimization import InternalBO
from partxv2.gprInterface import InternalGPR

def internal_function(X):
    return -(1.4 - 3.0 * X[0]) * np.sin(18.0 * X[0])

BENCHMARK_NAME = "Testing_12"
init_reg_sup = np.array([[0., 1.]])
tf_dim = 1
max_budget = 500
init_budget = 10
bo_budget = 10
cs_budget = 10
alpha = 0.05
R = 10
M = 100
delta = 0.001
fv_quantiles_for_gp = [0.01, 0.05, 0.5]
branching_factor = 2
uniform_partitioning = True
start_seed = 123
gpr_model = InternalGPR()
bo_model = InternalBO()

init_sampling_type = "lhs_sampling"
cs_sampling_type = "lhs_sampling"
q_estim_sampling = "lhs_sampling"
mc_integral_sampling_type = "lhs_sampling"
results_sampling_type = "lhs_sampling"
results_at_confidence = 0.95

test_function = internal_function
num_macro_reps = 5
results_folder_name = "Testing"
num_cores = 10
x = run_partx(BENCHMARK_NAME, test_function, num_macro_reps, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model, 
                init_sampling_type, cs_sampling_type, 
                q_estim_sampling, mc_integral_sampling_type, 
                results_sampling_type, 
                results_at_confidence, results_folder_name, num_cores)   


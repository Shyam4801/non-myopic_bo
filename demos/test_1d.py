import numpy as np
from bo import Behavior, PerformBO
from bo.bayesianOptimization import InternalBO
from bo.gprInterface import InternalGPR
from bo.interface import BOResult


def internal_function(X):
            return X[0] ** 2 + X[1] ** 2 -1

init_reg_sup = np.array([[-1, 1], [-2, 2]])

optimizer = PerformBO(
    test_function=internal_function,
    init_budget=10,
    max_budget=15,
    region_support=init_reg_sup,
    seed=12345,
    behavior=Behavior.MINIMIZATION,
    init_sampling_type="lhs_sampling"
)

z = optimizer(bo_model=InternalBO(), gpr_model=InternalGPR())
history = z.history
time = z.optimization_time

print(np.array(history, dtype=object))
print(f"Time taken to finish iterations: {round(time, 3)}s.")
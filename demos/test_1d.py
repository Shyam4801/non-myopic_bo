import numpy as np
from bo import Behavior, PerformBO
from bo.bayesianOptimization import InternalBO
from bo.bayesianOptimization import Rollout_BO
from bo.gprInterface import InternalGPR
from bo.interface import BOResult
import time

# def internal_function(X):
#     return -(1.4 - 3.0 * X[0]) * np.sin(18.0 * X[0])

# def internal_function(X):  # hm
#     return (X[0]**2 + X[1] - 11)**2 + (X[0] + X[1]**2 -7)**2

# def internal_function(X): #lg all 5 pts close to glob min
#     return X[0]**2 + X[1]**2 - 2*np.exp(-(X[0]**2+X[1]**2))

# def internal_function(X):
#     return X[0]**4 + X[1]**4 - 4*X[0]*X[1] + 1
 
# def internal_function(X):
#             # print('internal func :', X,X.shape)
#             return X[0] ** 2 + X[1] ** 2 -1

# def internal_function(X): # 1D to 2D
#     return (-(1.4 - 3.0 * X[0]) * np.sin(15.0 * X[0]))* X[1]

def internal_function(X): #Branin with unique glob min -  9.42, 2.475
        # if X.shape[1] != 2:
        #     raise Exception('Dimension must be 2')
        # d = 2
        # if lb is None or ub is None:
        #     lb = np.full((d,), 0)
        #     ub = np.full((d,), 0)
        #     lb[0] = -5
        #     lb[1] = 0
        #     ub[0] = 10
        #     ub[1] = 15
        # x = from_unit_box(x, lb, ub)
        x1 = X[0]
        x2 = X[1]
        t = 1 / (8 * np.pi)
        s = 10
        r = 6
        c = 5 / np.pi
        b = 5.1 / (4 * np.pi ** 2)
        a = 1
        term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        term2 = s * (1 - t) * np.cos(x1)
        l1 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-12.27)**2))
        l2 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-2.275)**2))
        return term1 + term2 + s + l1 + l2

# def internal_function(X): # func in paper
#     return np.sin(20*X[0]) + 20*(X[0] - 0.3)**2

# init_reg_sup = np.array([[-1, 1], [-2, 2]])
# init_reg_sup = np.array([[0, 1.5]])
# init_reg_sup = np.array([[-5, 5], [-5, 5]]) #hm
# init_reg_sup = np.array([[-2, 2], [-2, 2]]) #lg
# init_reg_sup = np.array([[0, 1.5], [-2, 2]])
init_reg_sup = np.array([[-5, 10], [0, 15]]) # branin
# init_reg_sup = np.array([[-0.07, 0.8]])

print('time: ',int(time.time()))
optimizer = PerformBO(
    test_function=internal_function,
    init_budget=2,  #10
    max_budget=12,  #15
    region_support=init_reg_sup,
    seed=int(time.time()),
    behavior=Behavior.MINIMIZATION,
    init_sampling_type="lhs_sampling"
)

z = optimizer(bo_model=Rollout_BO(2), gpr_model=InternalGPR())
history = z.history
time = z.optimization_time

print(np.array(history, dtype=object))
print(f"Time taken to finish iterations: {round(time, 3)}s.")
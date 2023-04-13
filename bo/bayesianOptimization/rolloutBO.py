from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
import copy
import time 

from .internalBO import InternalBO
from ..gprInterface import GPR
from ..sampling import uniform_sampling
from ..utils import compute_robustness
from ..behavior import Behavior
from .rolloutEI import RolloutEI



class Rollout_BO():
    def __init__(self,horizon):
        super().__init__()
        self.horizon = horizon
        

    def sample(
        self,
        test_function: Callable,
        num_samples: int,
        x_train: NDArray,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: Callable,
        rng,
        behavior:Behavior = Behavior.MINIMIZATION
    ) -> Tuple[NDArray]:

        """Internal BO Model

        Args:
            test_function: Function of System Under Test.
            num_samples: Number of samples to generate from BO.
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            rng: RNG object from numpy

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array or does not match dimensions
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train

        Returns:
            x_complete
            y_complete
            x_new
            y_new
        """
        falsified = False
        # generate max budget - init samples
        for sample in tqdm(range(num_samples)):
            model = GPR(gpr_model)
            model.fit(x_train, y_train)
            # Sample the next point by rollout for h horizons
            ei_roll = RolloutEI()
            pred_sample_x = ei_roll.sample(test_function,self.horizon,y_train,region_support,model,rng)
            pred_sample_y, falsified = compute_robustness(pred_sample_x, test_function, behavior)
            print('############################################################################')
            print()
            print('ypred: ',pred_sample_y,'xred BO: ',pred_sample_x)
            print()
            print('############################################################################')
            x_train = np.vstack((x_train, pred_sample_x))
            y_train = np.hstack((y_train, pred_sample_y))

            if falsified:
                break
        return falsified
    
    











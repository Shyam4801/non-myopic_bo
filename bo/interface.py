import enum
import time
import numpy as np
from attr import frozen
from typing import Callable, Any
from numpy.typing import NDArray
import pandas as pd
from .bayesianOptimization.viz import plot_1d,plot_obj

from .bayesianOptimization import BOSampling
from .utils import Fn, compute_robustness
from .sampling import uniform_sampling, lhs_sampling
from .behavior import Behavior
from .bayesianOptimization.constants import NAME,H

@frozen(slots=True)
class BOResult:
    """Data class that represents the result of a uniform random optimization.

    Attributes:
        average_cost: The average cost of all the samples selected.
    """

    history: Any
    optimization_time: float


class PerformBO:
    def __init__(
            self, 
            test_function: Callable,
            init_budget: int,
            max_budget: int,
            region_support: NDArray,
            seed,
            behavior:Behavior = Behavior.MINIMIZATION,
            init_sampling_type = "lhs_sampling",
        ):
            """Internal BO Model

            Args:
                test_function: Function of System Under Test.
                init_budget: Number of samples in Initial Budget,
                max_budget: Maximum budget
                region_support: Min and Max of all dimensions
                seed: Set seed for replicating Experiment
                behavior: Set behavior to Behavior.MINIMIZATION or Behavior.FALSIFICATION
                init_sampling_type: Choose between "lhs_sampling" or "uniform_sampling"

            Returns:
                x_complete
                y_complete
                x_new
                y_new
            """
            
            if max_budget < init_budget:
                raise ValueError("Max Budget cannot be less than Initial Budget")

            self.tf_wrapper = Fn(test_function)
            self.init_budget = init_budget
            self.max_budget = max_budget
            self.region_support = region_support
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.init_sampling_type = init_sampling_type
            self.behavior = behavior
            self.tf = test_function # changed

    def __call__(self, bo_model, gpr_model):
        start_time = time.perf_counter()
        tf_dim = self.region_support.shape[0]
        bo_routine = BOSampling(bo_model)
        
        
        if self.init_sampling_type == "lhs_sampling":
            x_train = np.array([[-1.425914942388530,7.7740223781336100],[2.897079800559	,10.561556213666200],[7.325439816266640,	1.3057551677419700],[4.859328387159200	,13.26614441528690],[-3.0007302930792900,	5.277732413490960]]) #lhs_sampling(self.init_budget, self.region_support, tf_dim, self.rng)
        elif self.init_sampling_type == "uniform_sampling":
            x_train = uniform_sampling(self.init_budget, self.region_support, tf_dim, self.rng)
        else:
            raise ValueError(f"{self.init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")

        y_train, falsified = compute_robustness(x_train, self.tf_wrapper, self.behavior)
        
        if not falsified:
            print("No falsification in Initial Samples. Performing BO now")
            falsified = bo_routine.sample(self.tf_wrapper, self.max_budget - self.init_budget, x_train, y_train, self.region_support, gpr_model, self.rng)

        df = pd.DataFrame(self.tf_wrapper.point_history)
        # df = df.iloc[:,1].apply(lambda x: x[0])
        print(df)
        xcoord = pd.DataFrame(df.iloc[:,1].to_list())
        xcoord['y'] = df.iloc[:,2]
        xcoord.to_csv(str(H)+'_'+NAME+'_'+str(self.init_budget)+'_'+str(self.max_budget - self.init_budget)+'.csv')
        xcoord = xcoord.to_numpy()
        print(xcoord)
        # plot_1d(xcoord,self.tf,0.25,0.07,0.8,self.init_budget,self.max_budget - self.init_budget)
        plot_obj(xcoord,self.tf,[9.42, 2.475],[-5,10],[0,15],self.init_budget,self.max_budget - self.init_budget)
        return BOResult(self.tf_wrapper.point_history, time.perf_counter()-start_time)
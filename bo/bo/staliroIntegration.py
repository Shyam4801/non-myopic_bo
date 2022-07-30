from dataclasses import dataclass
from typing import Any, List, Sequence, Callable

import numpy as np
from staliro.core import Interval, Optimizer, ObjectiveFn, Sample

from .interface import PerformBO

Bounds = Sequence[Interval]
BOResult = List[Any]

@dataclass(frozen=True)
class BO(Optimizer[BOResult]):
    """The PartX optimizer provides statistical guarantees about the existence of falsifying behaviour in a system."""

    benchmark_name:str
    init_budget: int
    gpr_model: Callable
    bo_model : Callable
    folder_name:str
    init_sampling_type:str

    def optimize(self, func: ObjectiveFn, bounds: Bounds, budget:int, seed: int) -> BOResult:
        region_support = np.array((tuple(bound.astuple() for bound in bounds),))[0]
        
        print("************************************************************************")
        print("************************************************************************")
        print("************************************************************************")
        print(f"Test Function:\n Testing function is a {region_support.shape[0]}d problem with initial region support of {region_support}.")
        print(f"Starting macro replications with seed {seed} and maximum budget of {budget}, where")
        print(f"Initilization Budget = {self.init_budget},\nBO Budget = {budget-self.init_budget},\n")
        print(f"Sampling Types\n-----------")
        print(f"init_sampling_type = {self.init_sampling_type}")
        print("************************************************************************")
        print("************************************************************************")
        print("************************************************************************")
        
        
        def test_function(sample: np.ndarray) -> float:
            return func.eval_sample(Sample(sample))
        
        bo = PerformBO(
            test_function = test_function,
            benchmark_name=self.benchmark_name,
            init_budget=self.init_budget,
            max_budget=budget,
            region_support=region_support,
            gpr_model=self.gpr_model,
            seed = seed,
            folder_name=self.folder_name,
            init_sampling_type=self.init_sampling_type
        )
        return bo(self.bo_model, self.gpr_model)
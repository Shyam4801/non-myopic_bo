
from AT_benchmark.AT_specifications import load_specification_dict
from models import AutotransModel
from Benchmark import Benchmark

from bo import BO
from bo.gprInterface import InternalGPR
from bo.bayesianOptimization import InternalBO

from staliro.staliro import staliro
from staliro.options import Options

# Define Signals and Specification
class Benchmark_AT63(Benchmark):
    def __init__(self, benchmark, results_folder) -> None:
        if benchmark != "AT63":
            raise ValueError("Inappropriate Benchmark name")

        self.results_folder = results_folder
        self.specification, self.signals = load_specification_dict(benchmark)
        print(self.specification)
        print(self.signals)
        self.MAX_BUDGET = 2000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        self.model = AutotransModel()
        
        self.optimizer = BO(
            benchmark_name=f"{benchmark}_budget_{self.MAX_BUDGET}_{self.NUMBER_OF_MACRO_REPLICATIONS}_reps",
            init_budget = 100,
            gpr_model = InternalGPR(),
            bo_model = InternalBO(),
            folder_name = results_folder,
            init_sampling_type = "lhs_sampling"
        )

        self.options = Options(runs=self.NUMBER_OF_MACRO_REPLICATIONS, iterations=self.MAX_BUDGET, interval=(0, 50), seed = 12345, signals=self.signals)

    def run(self):
        result = staliro(self.model, self.specification, self.optimizer, self.options)
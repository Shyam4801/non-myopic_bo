
from NN_benchmark.NN_specifications import load_specification_dict
from models import NNModel
from Benchmark import Benchmark
from seqsampling import run_UR

from staliro.staliro import staliro
from staliro.options import Options

# Define Signals and Specification
class Benchmark_NNx(Benchmark):
    def __init__(self, benchmark, results_folder) -> None:
        if benchmark != "NNx":
            raise ValueError("Inappropriate Benchmark name")

        self.results_folder = results_folder
        self.specification, self.initial_conditions = load_specification_dict(benchmark)
        
        self.MAX_BUDGET = 3000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        self.model = NNModel(0.005, 0.03)
        self.optimizer = run_UR(
            BENCHMARK_NAME=f"{benchmark}_budget_{self.MAX_BUDGET}_{self.NUMBER_OF_MACRO_REPLICATIONS}_reps",
            results_folder_name = results_folder,
        )

        self.options = Options(runs=self.NUMBER_OF_MACRO_REPLICATIONS, iterations=self.MAX_BUDGET, interval=(0, 3), seed = 12345, static_parameters = self.initial_conditions, signals=[])

    def run(self):
        result = staliro(self.model, self.specification, self.optimizer, self.options)
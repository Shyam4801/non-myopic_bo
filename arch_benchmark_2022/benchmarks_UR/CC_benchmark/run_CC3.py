
from CC_benchmark.CC_specifications import load_specification_dict
from models import CCModel
from Benchmark import Benchmark
from seqsampling import run_UR

from staliro.staliro import staliro
from staliro.options import Options

# Define Signals and Specification
class Benchmark_CC3(Benchmark):
    def __init__(self, benchmark, results_folder) -> None:
        if benchmark != "CC3":
            raise ValueError("Inappropriate Benchmark name")

        self.results_folder = results_folder
        self.specification, self.signals = load_specification_dict(benchmark)
        
        self.MAX_BUDGET = 3000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        self.model = CCModel()
        self.optimizer = run_UR(
            BENCHMARK_NAME=f"{benchmark}_budget_{self.MAX_BUDGET}_{self.NUMBER_OF_MACRO_REPLICATIONS}_reps",
            results_folder_name = results_folder,
        )

        self.options = Options(runs=self.NUMBER_OF_MACRO_REPLICATIONS, iterations=self.MAX_BUDGET, interval=(0, 100),  seed = 12345, signals=self.signals)

    def run(self):
        result = staliro(self.model, self.specification, self.optimizer, self.options)
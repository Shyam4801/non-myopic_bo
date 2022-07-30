import numpy as np
from models import SCModel
from Benchmark import Benchmark
from seqsampling import run_UR

from staliro.staliro import staliro
from staliro.options import Options, SignalOptions
from staliro.specifications import RTAMTDense
from staliro.signals import piecewise_constant

# Define Signals and Specification
class Benchmark_SC1(Benchmark):
    def __init__(self, benchmark, results_folder) -> None:
        if benchmark != "SC1":
            raise ValueError("Inappropriate Benchmark name")

        self.results_folder = results_folder
        phi = "G[30,35] ((pressure >= 87) and (pressure<=87.5))"
        self.specification = RTAMTDense(phi, {"pressure":0})
        self.signals = [
            SignalOptions(control_points = [(3.99, 4.01)]*20, signal_times=np.linspace(0.,35.,20, endpoint = False), factory=piecewise_constant),
        ]
        
        self.model = SCModel()
        self.MAX_BUDGET = 3000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        
        self.optimizer = run_UR(
            BENCHMARK_NAME=f"{benchmark}_budget_{self.MAX_BUDGET}_{self.NUMBER_OF_MACRO_REPLICATIONS}_reps",
            results_folder_name = results_folder,
        )

        self.options = Options(runs=self.NUMBER_OF_MACRO_REPLICATIONS, iterations=self.MAX_BUDGET, interval=(0, 35),  signals=self.signals)

    def run(self):
        result = staliro(self.model, self.specification, self.optimizer, self.options)
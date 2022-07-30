from collections import OrderedDict
from math import pi
import numpy as np

from models import F16Model
from Benchmark import Benchmark
from seqsampling import run_UR

from staliro.staliro import staliro
from staliro.options import Options
from staliro.specifications import RTAMTDense



F16_PARAM_MAP = OrderedDict({
    'air_speed': {
        'enabled': False,
        'default': 540
    },
    'angle_of_attack': {
        'enabled': False,
        'default': np.deg2rad(2.1215)
    },
    'angle_of_sideslip': {
        'enabled': False,
        'default': 0
    },
    'roll': {
        'enabled': True,
        'default': None,
        'range': (pi / 4) + np.array((-pi / 20, pi / 30)),
    },
    'pitch': {
        'enabled': True,
        'default': None,
        'range': (-pi / 2) * 0.8 + np.array((0, pi / 20)),
    },
    'yaw': {
        'enabled': True,
        'default': None,
        'range': (-pi / 4) + np.array((-pi / 8, pi / 8)),
    },
    'roll_rate': {
        'enabled': False,
        'default': 0
    },
    'pitch_rate': {
        'enabled': False,
        'default': 0
    },
    'yaw_rate': {
        'enabled': False,
        'default': 0
    },
    'northward_displacement': {
        'enabled': False,
        'default': 0
    },
    'eastward_displacement': {
        'enabled': False,
        'default': 0
    },
    'altitude': {
        'enabled': False,
        'default': 4040.0
    },
    'engine_power_lag': {
        'enabled': False,
        'default': 9
    }
})

class Benchmark_F16_alt4040(Benchmark):
    def __init__(self, benchmark, results_folder) -> None:
        if benchmark != "F16_alt4040":
            raise ValueError("Inappropriate Benchmark name")

        self.results_folder = results_folder

        self.model = F16Model(F16_PARAM_MAP)
        
        phi = "G[0,15] altitude > 0"
        self.specification = RTAMTDense(phi, {"altitude":0})
        self.initial_conditions = self.model.get_static_params()
        
        self.MAX_BUDGET = 3000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        
        self.optimizer = run_UR(
            BENCHMARK_NAME=f"{benchmark}_budget_{self.MAX_BUDGET}_{self.NUMBER_OF_MACRO_REPLICATIONS}_reps",
            results_folder_name = results_folder,
        )

        self.options = Options(runs=self.NUMBER_OF_MACRO_REPLICATIONS, iterations=self.MAX_BUDGET, interval=(0, 15),  static_parameters = self.initial_conditions, signals=[])

    def run(self):
        result = staliro(self.model, self.specification, self.optimizer, self.options)
        
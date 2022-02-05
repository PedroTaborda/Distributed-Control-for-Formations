from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from dynamics.car import CarParameters
from control.car_controller import ControllerParameters

@dataclass
class SimSettings:
    controller_sample_time: float = 0.1
    time_sim: float = 10.0

    cars_params: tuple[tuple[CarParameters, ControllerParameters]] = ((
            CarParameters(),
            ControllerParameters()
        ),)

    leader_pos: Callable[[float], float] = lambda t: t

@dataclass
class SimData:
    settings: SimSettings

    n_cars: int

    time: np.ndarray

    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    control_signals: np.ndarray
    references: np.ndarray

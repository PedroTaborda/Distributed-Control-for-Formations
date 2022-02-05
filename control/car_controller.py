from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

@dataclass
class ControllerParameters:
    """Defines the settings for the controller.
    """
    k: float = 1.0 # TODO: PLACEHOLDER
    
    d0: float = 1.0
    th: float = 0.0
    d: Callable[[float], float] = lambda v, th=th, d0=d0: th*v + d0 # desired distance between itself and the next car

class Controller:
    def __init__(self, params: ControllerParameters) -> None:
        self.params: ControllerParameters = params

    def control_input(self, x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Computes the control input for the given state and control input.
        """
        ref = x[0] + ref[0] - ref[1]
        return (ref - x[0] - x[1]) * self.params.k # TODO: PLACEHOLDER


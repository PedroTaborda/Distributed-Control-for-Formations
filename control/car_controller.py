from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class ControllerParameters:
    """Defines the settings for the controller.
    """
    k: float = 1.0 # TODO: PLACEHOLDER

class Controller:
    def __init__(self, params: ControllerParameters) -> None:
        self.params: ControllerParameters = params

    def control_input(self, x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Computes the control input for the given state and control input.
        """
        return (ref - x[0] - x[1]) * self.params.k # TODO: PLACEHOLDER


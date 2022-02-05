from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from control.controlled_car import ControlledCar

@dataclass
class DistributedControllerParameters:
    """Defines the settings for the distributed controller.
    """
    d0: float = 1.0
    th: float = 0.0
    d: Callable[[int, float], float] = lambda i, v, th=th, d0=d0: th*v + d0 # desired distance between cars i and i+1


class DistributedControlledSystem:
    """Consider platoon of controller cars (nodes) - ignoring information flow
    topology - and control them such that desired spacing between vehicles is 
    maintained, and they follow a leader
    """
    def __init__(self, params: DistributedControllerParameters, cars: tuple[ControlledCar]) -> None:
        self.params: DistributedControllerParameters = params
        self.cars = cars

        self.references = []

    def step(self, ref: float, time_step: float):
        """Steps entire system by time_step, using reference ref as the position
        of the leader of the platoon
        """
        cur_references = []

        for i, car in enumerate(self.cars):
            # compute desired distance to the next car
            d = self.params.d(i, car.state[1])

            if i == 0:
                pos_ref = ref - d
            else:
                pos_ref = self.cars[i-1].state[0] - d
            cur_references.append(pos_ref)

            car.step(pos_ref, time_step)

        self.references.append(cur_references)

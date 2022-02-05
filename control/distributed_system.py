from __future__ import annotations

import numpy as np

from control.controlled_car import ControlledCar

class DistributedControlledSystem:
    """Consider platoon of controller cars (nodes) and handle virtual sensor data
    (distance to car ahead, distance to car behind)
    """
    def __init__(self, cars: tuple[ControlledCar]) -> None:
        self.cars = cars

        self.references = []

    def step(self, leader_pos: float, time_step: float):
        """Steps entire system by time_step, using reference ref as the position
        of the leader of the platoon
        """
        cur_references = []

        for i, car in enumerate(self.cars):
            # compute desired distance to the next car
            current_car_front_pos = car.state[0]
            current_car_back_pos = car.state[0] - car.car.params.length

            if i == 0:
                car_ahead_back = leader_pos
                car_behind_front = self.cars[i+1].state[0]
            elif i == len(self.cars) - 1:
                car_ahead_back = self.cars[i-1].state[0]
                car_behind_front = np.inf
            else:
                car_ahead_back = self.cars[i-1].state[0]
                car_behind_front = self.cars[i+1].state[0]

            info = np.array([
                np.abs(car_ahead_back - current_car_front_pos),
                np.abs(car_behind_front - current_car_back_pos)
            ])
            
            cur_references.append(info)

            car.step(info, time_step)

        self.references.append(cur_references)

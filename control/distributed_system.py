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

    def step(self, leader_state: np.ndarray, time_step: float):
        """Steps entire system by time_step, using reference ref as the position
        of the leader of the platoon
        """
        cur_references = []


        for i, car in enumerate(self.cars):
            if i == 0:
                car_ahead_back_state = leader_state
                if len(self.cars) < 2:
                    car_behind_front_state = [-np.inf, 0, 0]
                else:
                    car_behind_front_state = self.cars[i+1].state
                    car_behind_front_state[0] = car_behind_front_state[0] + self.cars[i+1].car.params.length
            elif i == len(self.cars) - 1:
                car_ahead_back_state = self.cars[i-1].state
                car_behind_front_state = [-np.inf, 0, 0]
            else:
                car_ahead_back_state = self.cars[i-1].state
                car_behind_front_state = self.cars[i+1].state
                car_behind_front_state[0] = car_behind_front_state[0] + self.cars[i+1].car.params.length

            # convert to distances, as absolute position is not known
            car_ahead_back_state[0] = car_ahead_back_state[0] - (car.state[0] + car.car.params.length)
            car_behind_front_state[0] = car_behind_front_state[0] - car.state[0]

            info = np.array([
                car_ahead_back_state,
                car_behind_front_state
            ])
            
            cur_references.append(info)

            car.step(info, time_step)

        self.references.append(cur_references)

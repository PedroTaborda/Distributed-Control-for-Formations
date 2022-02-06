from __future__ import annotations
import copy

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
                car_ahead_back_state_pos = leader_state
            else:
                car_ahead_back_state_pos = self.cars[i-1].state


            # convert to distances, as absolute position is not known
            car_ahead_back_state_dist = copy.copy(car_ahead_back_state_pos)
            car_ahead_back_state_dist[0] = car_ahead_back_state_pos[0] - (car.state[0] + car.car.params.length)

            cur_references.append(car_ahead_back_state_dist)

            print(f"car_ahead_back_state_dist: {car_ahead_back_state_dist}")
            car.step(car_ahead_back_state_dist, time_step)

        self.references.append(cur_references)

from __future__ import annotations
import copy

import numpy as np
import multiprocessing as mp

from control.controlled_car import ControlledCar

class DistributedControlledSystem:
    """Consider platoon of controller cars (nodes) and handle virtual sensor data
    (distance to car ahead, distance to car behind)
    """
    def __init__(self, cars: tuple[ControlledCar]) -> None:
        self.cars = cars

        self.references = []
        self.control_signals = []

        self.positions = []
        self.velocities = []
        self.accelerations = []

    def init_sim_vars(self) -> None:
        self.n_jobs = len(self.cars)
        self.pool = mp.Pool(processes=self.n_jobs)

    @staticmethod
    def _step_func(cars: tuple[ControlledCar], i: int, leader_state: np.ndarray, time_step: float):
        car = cars[i]
        if i == 0:
            car_ahead_back_state_pos = leader_state
        else:
            car_ahead_back_state_pos = cars[i-1].state

        # convert to distances, as absolute position is not known
        car_ahead_back_state_dist = copy.copy(car_ahead_back_state_pos)
        car_ahead_back_state_dist[0] = car_ahead_back_state_pos[0] - (car.state[0] + car.car.params.length)

        car.step(car_ahead_back_state_dist, time_step)

        reference = car_ahead_back_state_pos[0] - (car.controller.params.d(car.state[1]) + car.car.params.length)

        return car.state, reference, car.control_signal

    def step(self, leader_state: np.ndarray, time_step: float):
        """Steps entire system by time_step, using reference ref as the position
        of the leader of the platoon
        """
        cur_references = []
        cur_control_signals = []

        # step all cars using multiprocessing
        #with self.pool as pool:
        states_refs_control = self.pool.starmap(
            self._step_func,
            zip(
                [self.cars] * self.n_jobs,
                range(self.n_jobs),
                [leader_state] * self.n_jobs,
                [time_step] * self.n_jobs
            )
        )
        
        for i, car in enumerate(self.cars):
            state, reference, control_signal = states_refs_control[i]
            car.state = state

            cur_references.append(reference)
            cur_control_signals.append(control_signal)


        self.positions.append(np.array([car.state[0] for car in self.cars]))
        self.velocities.append(np.array([car.state[1] for car in self.cars]))
        self.accelerations.append(np.array([car.state[2] for car in self.cars]))

        self.references.append(cur_references)
        self.control_signals.append(cur_control_signals)

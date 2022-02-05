from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from dynamics.car import CarParameters, Car
from control.car_controller import ControllerParameters, Controller


class ControlledCar:
    """Connects a car to a controller, assuming continuous time car dynamics and
    controller.
    """
    def __init__(self, car_params: CarParameters, controller_params: ControllerParameters):
        self.car = Car(car_params)
        self.controller = Controller(controller_params)

        # define initial state
        self.state = np.array([
            self.car.params.pos_i, 
            self.car.params.vel_i, 
            self.car.params.acc_i
        ])

        self.control_signals = []
        self.states = [self.state]

    def _step_func(self, t: float, x: np.ndarray, u: float) -> np.ndarray:
        return self.car.state_space_dynamics(x, u) 

    def step(self, ref: float, time_step: float) -> np.ndarray:
        u = self.controller.control_input(self.state, ref)
        self.state = np.array(solve_ivp(self._step_func, [0, time_step], self.state, args=(u,)).y[:, -1])

        self.control_signals.append(self.controller.control_input(self.state, ref))
        self.states.append(self.state)

        return self.state

if __name__ == "__main__":
    # test
    car_params = CarParameters()
    controller_params = ControllerParameters()

    car = ControlledCar(car_params, controller_params)

    for i in range(100):
        car.step(1, 0.1)
        print(car.states[-1])
        print(car.control_signals[-1])
    

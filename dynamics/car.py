from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class CarParameters:
    """Defines the settings for the car.
    """
    length: float = 3.5     # length in meters

    wheel_radius: float = 0.256  # wheel radius in meters
    powertrain_efficiency: float = 0.8  # powertrain efficiency in [0, 1]

    drag_coefficient: float = 0.5  # drag coefficient in kg/m
    mass: float = 810       # mass in kg

    gravity: float = 9.81  # gravity in m/s^2
    rolling_resistance: float = 0.01  # rolling resistance in TODO: units (seems like meter)

    inertial_delay: float = 0.5  # delay in seconds

    pos_i: float = 0.0  # initial position in meters
    vel_i: float = 0.0  # initial velocity in meters/second
    T_i: float = 0.0  # initial acceleration in meters/second^2


class Car:
    """Models dynamics of a single car (time invariant)
    """

    def __init__(self, params: CarParameters) -> None:
        self.params: CarParameters = params

        # pre compute some coefficients so as not to clutter the state space
        self.vdot_coefs = np.array([
            params.powertrain_efficiency/(params.mass*params.wheel_radius),
            params.drag_coefficient/params.mass,
            params.rolling_resistance*params.gravity
        ])

    def state_pva(self, x: np.ndarray) -> np.ndarray:
        """Computes the position, velocity and acceleration of the car.
        """
        return np.array([x[0], x[1], np.dot(self.vdot_coefs, [x[2], x[1], 1])])

    def state_space_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes the derivative of the state vector, given the current state 
        and the control input.
        """
        derivative_pos = x[1]

        derivative_vel = np.dot(self.vdot_coefs, [x[2], x[1], 1])

        derivative_T = -(1.0/self.params.inertial_delay)*(x[2] - u)

        return np.array([derivative_pos, derivative_vel, derivative_T])

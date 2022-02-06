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

    deadzone_threshold_velocity: float = 0.1  # velocity in m/s

    inertial_delay: float = 0.1  # delay in seconds

    pos_i: float = 0.0  # initial position in meters
    vel_i: float = 0.0  # initial velocity in meters/second
    accel_i: float = 0.0  # initial acceleration in meters/second^2

def deadzone(param: float, threshold: float, continuous: bool = True):
    """ Implements a deadzone on a given input.
    """
    if abs(param) < threshold:
        return 0
    else:
        return param - threshold * np.sign(param) if continuous else param

class Car:
    """Models dynamics of a single car (time invariant)
    """

    def __init__(self, params: CarParameters) -> None:
        self.params: CarParameters = params

        # pre compute some coefficients
        # they multiply with u, a, v*v, a*v and 1
        self.adot_coefs = np.array([
            params.powertrain_efficiency/(params.mass*params.wheel_radius*params.inertial_delay),
            -1/params.inertial_delay,
            -params.drag_coefficient/(params.mass*params.inertial_delay),
            -2*params.drag_coefficient/params.mass,
            -params.rolling_resistance*params.gravity/params.inertial_delay
        ])

    # def state_pva(self, x: np.ndarray) -> np.ndarray:
    #     """Computes the position, velocity and acceleration of the car.
    #     """
    #     return np.array([x[0], x[1], np.dot(self.adot_coefs, [x[2], x[1], 1])])

    def state_space_dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        """Computes the derivative of the state vector, given the current state
        and the control input.
        """
        v = x[1]
        a = x[2]
        derivative_pos = v

        derivative_vel = a

        derivative_accel = np.dot(self.adot_coefs, [u, a, v*np.abs(v), a*v, 1*np.sign(deadzone(v, self.params.deadzone_threshold_velocity))])

        return np.array([derivative_pos, derivative_vel, derivative_accel])

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from dynamics.car import Car, CarParameters

@dataclass
class ControllerParameters:
    """Defines the settings for the controller.
    """
    k: float = 1.0 # TODO: PLACEHOLDER
    
    d0: float = 1.0
    th: float = 0.0
    d: Callable[[float], float] = lambda v, th=th, d0=d0: th*v + d0 # desired distance between itself and the next car

    car_params: CarParameters = CarParameters()
    mpc_t_horizon: float = 1.0
    mpc_n_horizon: int = 10

    mpc_step_size: float = mpc_t_horizon / mpc_n_horizon

    mpc_u_min: float = -10000.0
    mpc_u_max: float = 10000.0

    mpc_u_weight_factor: float = 1.0

class Controller:
    def __init__(self, params: ControllerParameters) -> None:
        self.params: ControllerParameters = params
        self.car: Car = Car(params.car_params)

        self.U0 = np.zeros(self.params.mpc_n_horizon)

    def _mpc_cost_fcn(self, x: np.ndarray, U: np.ndarray, car_ahead_states: np.ndarray, car_behind_states: np.ndarray) -> np.ndarray:
        """Computes the cost function for the MPC problem.
        """
        # simulate car dynamics in time_vec for the given control input
        self_states = np.zeros((self.params.mpc_n_horizon, 3))
        car_ivp_friendly_state_space = lambda t, x, u: self.car.state_space_dynamics(x, u)
        for i in range(0, self.params.mpc_n_horizon):
            if i == 0:
                self_states[i, :] = np.array(solve_ivp(car_ivp_friendly_state_space, [0, self.params.mpc_step_size], x, args=(U[i-1],)).y[:, -1])
            else:
                self_states[i, :] = np.array(solve_ivp(car_ivp_friendly_state_space, [0, self.params.mpc_step_size], self_states[i-1], args=(U[i-1],)).y[:, -1])
        
        distances_to_front = car_ahead_states[:, 0] - (self_states[:, 0] + self.car.params.length)
        reference_distances = np.array([self.params.d(state[1]) for state in car_ahead_states])

        errors = reference_distances - distances_to_front

        error_cost = np.sum(errors**2)

        control_cost = np.sum(U**2)

        return error_cost + self.params.mpc_u_weight_factor * control_cost

    def control_input(self, x: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Computes the control input for the given state and control input.
        """
        car_ahead_state = inputs[0]
        car_behind_state = inputs[1]
        # convert to absolute position in same reference frame as state to facilitate 
        # computation of errors
        car_ahead_state[0] = x[0] + car_ahead_state[0]
        car_behind_state[0] = x[0] + car_behind_state[0]

        time_vec = np.linspace(0, self.params.mpc_t_horizon, self.params.mpc_n_horizon, endpoint=False)
        t_step = time_vec[1] - time_vec[0]

        car_ahead_states = np.array([second_order_approx(car_ahead_state, t+self.params.mpc_step_size) for t in time_vec])
        car_behind_states = np.array([second_order_approx(car_behind_state, t+self.params.mpc_step_size) for t in time_vec])
        cost = lambda U: self._mpc_cost_fcn(x, U, car_ahead_states, car_behind_states)

        

        bnds = [(self.params.mpc_u_min, self.params.mpc_u_max)]*self.params.mpc_n_horizon
        # print(bnds)

        res = minimize(cost, np.zeros_like(self.U0), method="SLSQP", bounds=bnds, options={'disp': False, "maxiter": 100})

        if not res.success:
            print(res)
            raise Exception(f"MPC failed with error: {res.message}")

        self.U0 = res.x

        u_mpc = res.x[0]


        return u_mpc

def second_order_approx(init_state: np.ndarray, t: float) -> np.ndarray:
    """Computes a second order approximation of the car's state at time t.
    """
    return np.array([
        init_state[0] + init_state[1]*t + 0.5*init_state[2]*t**2, 
        init_state[1] + init_state[2]*t, 
        init_state[2]
    ])

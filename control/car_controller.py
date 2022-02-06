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
    k: float = 1.0  # TODO: PLACEHOLDER

    d0: float = 1.0
    th: float = 0.0
    d: Callable[[float], float] = lambda v, th=th, d0=d0: th*v + d0  # desired distance between itself and the next car

    car_params: CarParameters = CarParameters()
    mpc_t_horizon: float = 10.0
    mpc_n_horizon: int = 5

    mpc_step_size: float = mpc_t_horizon / mpc_n_horizon

    mpc_u_min: float = -10000.0
    mpc_u_max: float = 10000.0

    mpc_u_weight_factor: float = 1e-4


class Controller:
    def __init__(self, params: ControllerParameters) -> None:
        self.params: ControllerParameters = params
        self.car: Car = Car(params.car_params)

        self.U0 = np.ones(self.params.mpc_n_horizon*4)

    def _error_to_ref(self, PVAU: np.ndarray, car_ahead_states: np.ndarray, disp: bool = False) -> np.ndarray:
        P, V, A, U = self.unpack_PVAU(PVAU)
        distances_to_front = car_ahead_states[:, 0] - (P + self.car.params.length)
        reference_distances = np.array([self.params.d(state[1]) for state in car_ahead_states])

        if disp:
            print(f"Reference distances: {reference_distances}")
            print(f"Distances to front: {distances_to_front}")

        return reference_distances - distances_to_front

    def _mpc_cost_fcn(self, PVAU: np.ndarray, car_ahead_states: np.ndarray, disp: bool = False) -> np.ndarray:
        """Computes the cost function for the MPC problem.
        """
        P, V, A, U = self.unpack_PVAU(PVAU)
        errors = self._error_to_ref(PVAU, car_ahead_states, disp)

        error_cost = np.sum(errors**2)

        control_cost = np.sum(U**2)

        if disp:
            print(f"Errors: {errors}")
            print(f"Error cost: {error_cost}")
            print(f"Control cost: {self.params.mpc_u_weight_factor *control_cost}")

        return error_cost + self.params.mpc_u_weight_factor * control_cost

    def _mpc_cost_jac(self, PVAU: np.ndarray, car_ahead_states):
        P, V, A, U = self.unpack_PVAU(PVAU)
        distances_to_front = car_ahead_states[:, 0] - (P + self.car.params.length)
        reference_distances = np.array([self.params.d(state[1]) for state in car_ahead_states])

        errors = reference_distances - distances_to_front

        N = self.params.mpc_n_horizon
        matrix = np.block([
            [np.eye(N), np.zeros((N, 3*N))],
            [np.zeros((2*N, 4*N))],
            [np.zeros((N, 3*N)), self.params.mpc_u_weight_factor * np.eye(N)]
        ])
        PVAU[:N] = errors
        return 2 * matrix @ PVAU

    def _mpc_cost_hess(self):
        N = self.params.mpc_n_horizon
        matrix = np.block([
            [np.eye(N), np.zeros((N, 3*N))],
            [np.zeros((2*N, 4*N))],
            [np.zeros((N, 3*N)), self.params.mpc_u_weight_factor * np.eye(N)]
        ])
        return 2 * matrix

    def control_input(self, x: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Computes the control input for the given state and control input.
        """
        car_ahead_state_rel_pos = inputs
        # convert to absolute position in same reference frame as state to facilitate
        # computation of errors
        car_ahead_state = car_ahead_state_rel_pos + np.array([x[0] + self.car.params.length, 0.0, 0.0])

        time_vec = np.linspace(0, self.params.mpc_t_horizon, self.params.mpc_n_horizon, endpoint=False)
        t_step = time_vec[1] - time_vec[0]

        car_ahead_states = np.array([second_order_approx(
            car_ahead_state, t+self.params.mpc_step_size) for t in time_vec])

        N = self.params.mpc_n_horizon

        def derivative(x, u):
            derivative_state_sim = self.car.state_space_dynamics(x, u)
            return derivative_state_sim

        def dynamics_constraint(PVAU, i):
            P, V, A, U = self.unpack_PVAU(PVAU)
            X = np.block([[P], [V], [A]])

            if i == 0:
                dX = X[:, i] - x
            else:
                dX = X[:, i] - X[:, i-1]

            return dX - derivative(X[:, i], U[i])*self.params.mpc_step_size

        def collision_constraint(PVAU, car_ahead_state, i):
            P, V, A, U = self.unpack_PVAU(PVAU)

            return car_ahead_state[0] - P[i]  # ineq constraints are non-negative

        cons = ({'type': 'eq', 'fun': lambda PVAU, i=i: dynamics_constraint(PVAU, i)} for i in range(N))
        cons += ({'type': 'ineq', 'fun': lambda PVAU, i=i: collision_constraint(PVAU, i)} for i in range(N))

        def cost(PVAU): return self._mpc_cost_fcn(PVAU, car_ahead_states)
        def jac(PVAU): return self._mpc_cost_jac(PVAU, car_ahead_states)

        bnds = [(None, None)]*3*N + [(self.params.mpc_u_min, self.params.mpc_u_max)]*N
        # print(bnds)

        res = minimize(cost, self.U0, method="SLSQP", bounds=bnds, constraints=cons,
                       jac=jac, options={'disp': False, "maxiter": 100})

        if not res.success:
            print(res)
            # raise Exception(f"MPC failed with error: {res.message}")

        self.U0 = res.x

        u_mpc = res.x[3*N]
        # self._mpc_cost_fcn(res.x, car_ahead_states, disp=True)
        return u_mpc

    @ staticmethod
    def pack_pvau(X, U):
        P = X[:, 0]
        V = X[:, 1]
        A = X[:, 2]
        return np.concatenate((P, V, A, U), axis=0)

    def unpack_PVAU(self, PVAU):
        N = self.params.mpc_n_horizon
        P = PVAU[:N]
        V = PVAU[N:2*N]
        A = PVAU[2*N:3*N]
        U = PVAU[3*N:]
        return (P, V, A, U)


def second_order_approx(init_state: np.ndarray, t: float) -> np.ndarray:
    """Computes a second order approximation of the car's state at time t.
    """
    return np.array([
        init_state[0] + init_state[1]*t + 0.5*init_state[2]*t**2,
        init_state[1] + init_state[2]*t,
        init_state[2]
    ])

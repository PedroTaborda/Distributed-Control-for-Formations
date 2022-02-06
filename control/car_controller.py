from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from dynamics.car import Car, CarParameters

@dataclass
class ControllerParameters:
    """Defines the settings for the controller.
    """
    d0: float = 10.0
    th: float = 1.0

    def d(self, v: float): 
        return max(v, 0)*self.th + self.d0

    car_params: CarParameters = CarParameters()
    mpc_t_horizon: float = 5.0
    mpc_n_horizon: int = 5

    mpc_sim_steps_per_control_step: int = 10

    mpc_step_size: float = mpc_t_horizon / mpc_n_horizon

    mpc_u_min: float = -10000.0
    mpc_u_max: float = 10000.0

    mpc_u_weight_factor: float = 1e-13

    mpc_optimizer_max_iter: int = 100
    mpc_optimizer_ftol: float = 1e-6

    mpc_acceleration_factor_second_order_approx: float = 1


class Controller:
    def __init__(self, params: ControllerParameters) -> None:
        self.params: ControllerParameters = params
        self.car: Car = Car(params.car_params)

        self.U0 = np.ones(self.params.mpc_n_horizon*4)

    def inverse_leaky_relu(self, x, negative_penalty=10):
        return np.where(x > 0, x, x * negative_penalty)

    def _error_to_ref(self, PVAU: np.ndarray, car_ahead_states: np.ndarray) -> np.ndarray:
        P, V, A, U = self.unpack_PVAU(PVAU)
        distances_to_front = car_ahead_states[:, 0] - (P + self.car.params.length)
        reference_distances = np.array([self.params.d(state[1]) for state in car_ahead_states])

        return self.inverse_leaky_relu(reference_distances - distances_to_front)

    def _mpc_cost_fcn(self, PVAU: np.ndarray, car_ahead_states: np.ndarray, disp: bool = False) -> np.ndarray:
        """Computes the cost function for the MPC problem.
        """
        P, V, A, U = self.unpack_PVAU(PVAU)
        errors = self._error_to_ref(PVAU, car_ahead_states)

        error_cost = np.sum(errors**2)

        control_cost = np.sum(U**2)

        if disp:
            print(f"Errors: {errors}")
            print(f"Error cost: {error_cost}")
            print(f"Control cost: {self.params.mpc_u_weight_factor *control_cost}")

        return error_cost + self.params.mpc_u_weight_factor * control_cost

    def _mpc_cost_jac(self, PVAU: np.ndarray, car_ahead_states):
        errors = self._error_to_ref(PVAU, car_ahead_states)

        N = self.params.mpc_n_horizon
        matrix = np.block([
            [np.eye(N), np.zeros((N, 3*N))],
            [np.zeros((2*N, 4*N))],
            [np.zeros((N, 3*N)), self.params.mpc_u_weight_factor * np.eye(N)]
        ])
        PVAU[:N] = errors
        return 2 * matrix @ PVAU

    def control_input(self, x: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Computes the control input for the given state and control input.
        """
        car_ahead_state_rel_pos = inputs
        # convert to absolute position in same reference frame as state to facilitate
        # computation of errors
        car_ahead_state = car_ahead_state_rel_pos + np.array([x[0] + self.car.params.length, 0.0, 0.0])

        time_vec = np.linspace(0, self.params.mpc_t_horizon, self.params.mpc_n_horizon, endpoint=False)
        t_step = time_vec[1] - time_vec[0]

        car_ahead_states = np.array([
            second_order_approx(
                car_ahead_state, 
                t+self.params.mpc_step_size,
                acceleration_factor=self.params.mpc_acceleration_factor_second_order_approx
                ) 
                for t in time_vec
            ])

        N = self.params.mpc_n_horizon

        def derivative(t, x, u):
            derivative_state_sim = self.car.state_space_dynamics(x, u)
            return derivative_state_sim

        def simulate_for_step(state: np.ndarray, u: float, t_sim: float, n_steps: int):
            dt = t_sim / n_steps
            for _ in range(n_steps):
                state = state + derivative(t_sim, state, u)*dt
                t_sim += t_step
            return state

        def dynamics_constraint(PVAU, i):
            P, V, A, U = self.unpack_PVAU(PVAU)
            X = np.block([[P], [V], [A]])

            if i == 0:
                init_X = x
            else:
                init_X = X[:, i-1]
            
            return X[:, i] - simulate_for_step(init_X, U[i], self.params.mpc_step_size, self.params.mpc_sim_steps_per_control_step)
        
        def collision_constraint(PVAU, i):
            P, V, A, U = self.unpack_PVAU(PVAU)

            return car_ahead_states[i][0] - (P[i] + self.car.params.length)  # ineq constraints are non-negative

        def collision_constraint_jac(PVAU, i):
            zeros_vec = np.zeros_like(PVAU)
            zeros_vec[i] = -1.0
            return zeros_vec
        
        dynamics_scale: np.ndarray = np.array([1e-3, 1/50, 1/10])
        numeric_scale_factor: float = 1e-3*1000

        cons = list(({
            'type': 'eq', 
            'fun': lambda PVAU, i=i: dynamics_scale*dynamics_constraint(PVAU, i)
            } for i in range(N)
        ))
        
        cons += list(({
            'type': 'ineq', 
            'jac': lambda PVAU, i=i: numeric_scale_factor*collision_constraint_jac(PVAU, i),
            'fun': lambda PVAU, i=i: numeric_scale_factor*collision_constraint(PVAU, i)
            } for i in range(N)
        ))
        
        def cost(PVAU): return numeric_scale_factor*self._mpc_cost_fcn(PVAU, car_ahead_states)
        def jac(PVAU): return numeric_scale_factor*self._mpc_cost_jac(PVAU, car_ahead_states)

        bnds = [(None, None)]*3*N + [(self.params.mpc_u_min, self.params.mpc_u_max)]*N
        # print(bnds)

        res = minimize(
            cost, 
            self.U0, 
            method="SLSQP", 
            bounds=bnds, 
            constraints=cons,
            jac=jac, 
            options={
                'disp': False, 
                "maxiter": self.params.mpc_optimizer_max_iter,
                "ftol": self.params.mpc_optimizer_ftol,
            }
        )

        #print('contraint: ', collision_constraint(res.x, 0))

        #if not res.success:
        #    print(f"MPC failed with error: {res.message}")
        #    print(res)

        self.U0 = res.x

        u_mpc = res.x[3*N]

        #import matplotlib.pyplot as plt
        #plt.clf()
        #plt.plot(time_vec, res.x[:N], label="P")
        #plt.plot(time_vec, car_ahead_states[:, 0], label="P_ahead")
        #refs = car_ahead_states[:, 0] - np.array([self.params.d(state[1]) for state in car_ahead_states])
        #plt.plot(time_vec, refs, label="ref")
        #plt.legend()
        #plt.pause(0.001)

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


def second_order_approx(init_state: np.ndarray, t: float, acceleration_factor: float = 0.5) -> np.ndarray:
    """Computes a second order approximation of the car's state at time t.
    """
    a = init_state[2]*acceleration_factor
    v = init_state[1]
    p = init_state[0]
    return np.array([
        p + v*t + a*0.5*t**2,
        v + a*t,
        a
    ])

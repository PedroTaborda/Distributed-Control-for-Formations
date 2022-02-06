import matplotlib.pyplot as plt
import numpy as np

from simulation.sim_dataflow import SimSettings, SimData
from simulation.simulator import Simulator

from dynamics.car import CarParameters

from control.car_controller import ControllerParameters


from visualization.plot_primitives import plot_car_positions, plot_car_reference_error, plot_intercar_distances

if __name__ == "__main__":
    def leader_state(t):
        # leader velocity is a linear interpolation of the following points
        # a point is (t, v)
        v_points = ((0, 0), (1, 0), (6, 50), (15, 50), (21, 0))
        pos_initial = 0

        pos = pos_initial
        # Before the specified points, velocity is constant
        first_t, first_v = v_points[0]
        if t < first_t:
            accel=0
            vel=first_v
            pos += (t-first_t)*vel
            return np.array[(pos, vel, accel)]
        
        prev_time = first_t
        prev_vel = first_v
        for t_p, v_p in v_points[1:]:
            accel = (v_p - prev_vel)/(t_p - prev_time)
            if t <= t_p:
                delta_t = t - prev_time
                vel = prev_vel + delta_t*accel
                pos += delta_t*prev_vel + delta_t**2 * accel / 2
                return np.array([pos, vel, accel])
            else:
                pos += (t_p - prev_time)*(prev_vel + v_p)/2
            prev_time = t_p
            prev_vel = v_p
        # if code reaches here, then time is after all points
        final_t, final_v = v_points[-1]
        accel=0
        vel=final_v
        pos += (t-final_t)*vel
        return np.array([pos, vel, accel])

    # example simulation with three cars
    settings = SimSettings(
        controller_sample_time=0.1,
        time_sim=30.0,
        cars_params=(
            (
                CarParameters(pos_i=-10.0),
                ControllerParameters()
            ),
            (
                CarParameters(pos_i=-30.0),
                ControllerParameters()
            ),
            (
                CarParameters(pos_i=-50.0),
                ControllerParameters()
            )
        ),
        leader_state = leader_state # lambda t: np.array([t, 1, 0])  #
    )

    sim = Simulator(settings)

    try:
        sim.simulate()
    finally:
        data: SimData = sim.get_sim_data()

        plt.figure()
        plot_car_positions(data, include_leader=True)
        plt.show()

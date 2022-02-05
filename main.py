import matplotlib.pyplot as plt

from simulation.sim_dataflow import SimSettings, SimData
from simulation.simulator import Simulator

from dynamics.car import CarParameters

from control.car_controller import ControllerParameters


from visualization.plot_primitives import plot_car_positions, plot_car_reference_error, plot_intercar_distances

if __name__ == "__main__":
    def leader_pos(t):
        # leader velocity is a zero order hold of the following points
        # a point is (t, v)
        v_points = ((0, 0), (1, 50), (5, 0))
        pos_initial = 0

        pos = pos_initial
        # Before the specified points, velocity is 0
        prev_time = -3  # any value can be assigned here, irrelevant
        prev_vel = 0
        for t_p, v_p in v_points:
            if t <= t_p:
                return pos + (t-prev_time)*prev_vel
            else:
                pos += (t_p - prev_time)*prev_vel
            prev_time = t_p
            prev_vel = v_p
        # if code reaches here, then time is after all points
        final_t, final_v = v_points[-1]
        return pos + (t - final_t)*final_v

    # example simulation with three cars
    settings = SimSettings(
        controller_sample_time=0.1,
        time_sim=100.0,
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
        leader_pos=leader_pos
    )

    sim = Simulator(settings)

    sim.simulate()

    data: SimData = sim.get_sim_data()

    plt.figure()
    plot_car_positions(data, include_leader=True)
    plt.show()

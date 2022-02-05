import matplotlib.pyplot as plt

from simulation.sim_dataflow import SimSettings, SimData
from simulation.simulator import Simulator

from dynamics.car import CarParameters

from control.car_controller import ControllerParameters

from visualization.plot_primitives import plot_car_positions, plot_car_reference_error, plot_intercar_distances

if __name__ == "__main__":


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
        leader_pos=lambda t: t
    )

    sim = Simulator(settings)

    sim.simulate()

    data: SimData = sim.get_sim_data()

    plt.figure()
    plot_car_positions(data, include_leader=True)
    plt.show()
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from control.car_controller import ControllerParameters
from dynamics.car import CarParameters

from examples.leaders import slow_leader, avg_speed_leader, fast_leader
from simulation.sim_dataflow import SimSettings
from simulation.simulator import Simulator
from visualization.plot_primitives import plot_car_positions


def ex_avg_speed_leader(do_plots: bool = True, figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=fast_leader,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters()
        ) for pos in [-10, -20, -30, -40]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, 'ex_avg_speed_leader_positions.pdf'))

    plt.show(block=False)
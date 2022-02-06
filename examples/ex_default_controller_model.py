from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from control.car_controller import ControllerParameters
from dynamics.car import CarParameters
from examples.leaders import slow_leader, avg_speed_leader, fast_leader
from simulation.sim_dataflow import SimSettings
from simulation.simulator import Simulator
from visualization.plot_primitives import plot_car_positions

def ex_def_controller(leader: Callable = avg_speed_leader, do_plots: bool = True, fig_file: str = 'ex_avg_speed_leader.pdf', figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=leader,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters()
        ) for pos in [-13.5, -27.0, -40.5, -54.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, fig_file))

    plt.show(block=False)

def ex_avg_speed_leader(do_plots: bool = True, figs_dir: str = '') -> None:
    ex_def_controller(avg_speed_leader, do_plots=do_plots, figs_dir=figs_dir, fig_file='ex_avg_speed_leader.pdf')

def ex_slow_leader(do_plots: bool = True, figs_dir: str = '') -> None:
    ex_def_controller(slow_leader, do_plots=do_plots, figs_dir=figs_dir, fig_file='ex_slow_leader.pdf')

def ex_fast_leader(do_plots: bool = True, figs_dir: str = '') -> None:
    ex_def_controller(fast_leader, do_plots=do_plots, figs_dir=figs_dir, fig_file='ex_fast_leader.pdf')

def leader_immobile(t: float):
    return np.array([0, 0, 0])

def ex_def_controller_immobile_leader(do_plots: bool = True, figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=leader_immobile,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters()
        ) for pos in [-13.5, -27.0, -40.5, -54.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, 'ex_def_controller_immobile_leader.pdf'))

    plt.show(block=False)

def ex_def_controller_immobile_leader_transient(do_plots: bool = True, figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=leader_immobile,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters()
        ) for pos in [-30.0, -60.0, -70.0, -110.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, 'ex_def_controller_immobile_leader_transient.pdf'))

    plt.show(block=False)

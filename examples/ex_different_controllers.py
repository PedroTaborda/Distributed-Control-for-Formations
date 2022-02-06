from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt

from control.car_controller import ControllerParameters
from dynamics.car import CarParameters
from examples.leaders import slow_leader, avg_speed_leader, fast_leader
from simulation.sim_dataflow import SimSettings
from simulation.simulator import Simulator
from visualization.plot_primitives import plot_car_positions

def ex_avg_leader_mpc_horizon_10_sec_5_N(do_plots: bool = True, figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=avg_speed_leader,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters(
                mpc_t_horizon=10.0,
                mpc_n_horizon=5,
            )
        ) for pos in [-13.5, -27.0, -40.5, -54.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, 'ex_avg_leader_mpc_horizon_10_sec_5_N.pdf'))

    plt.show(block=False)

def ex_avg_leader_mpc_horizon_10_sec_10_N(do_plots: bool = True, figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=avg_speed_leader,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters(
                mpc_t_horizon=10.0,
                mpc_n_horizon=10,
            )
        ) for pos in [-13.5, -27.0, -40.5, -54.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, 'ex_avg_leader_mpc_horizon_10_sec_10_N.pdf'))

    plt.show(block=False)

def ex_avg_leader_mpc_horizon_1_sec_5_N(do_plots: bool = True, figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=avg_speed_leader,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters(
                mpc_t_horizon=1.0,
                mpc_n_horizon=5,
            )
        ) for pos in [-13.5, -27.0, -40.5, -54.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, 'ex_avg_leader_mpc_horizon_1_sec_5_N.pdf'))

    plt.show(block=False)

def ex_avg_leader_half_acceleration_prediction_constant_distance_keeping(do_plots: bool = True, figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=avg_speed_leader,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters(
                th=0.0,
                mpc_acceleration_factor_second_order_approx=0.5,
            )
        ) for pos in [-13.5, -27.0, -40.5, -54.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, 'ex_avg_leader_half_acceleration_prediction_constant_distance_keeping.pdf'))

    plt.show(block=False)

def ex_avg_leader_half_acceleration_prediction_usual_distance_keeping(do_plots: bool = True, figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=avg_speed_leader,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters(
                mpc_acceleration_factor_second_order_approx=0.5,
            )
        ) for pos in [-13.5, -27.0, -40.5, -54.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, 'ex_avg_leader_half_acceleration_prediction_usual_distance_keeping.pdf'))

    plt.show(block=False)

def ex_avg_leader_zero_acceleration_prediction_constant_distance_keeping(do_plots: bool = True, figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=avg_speed_leader,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters(
                th=0.0,
                mpc_acceleration_factor_second_order_approx=0.0,
            )
        ) for pos in [-13.5, -27.0, -40.5, -54.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, 'ex_avg_leader_zero_acceleration_prediction_constant_distance_keeping.pdf'))

    plt.show(block=False)

def ex_avg_leader_zero_acceleration_prediction_usual_distance_keeping(do_plots: bool = True, figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=avg_speed_leader,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters(
                mpc_acceleration_factor_second_order_approx=0.0,
            )
        ) for pos in [-13.5, -27.0, -40.5, -54.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, 'ex_avg_leader_zero_acceleration_prediction_usual_distance_keeping.pdf'))

    plt.show(block=False)

def ex_constant_dist_to_next_car(leader: Callable = avg_speed_leader, do_plots: bool = True, fig_file: str = 'ex_avg_speed_leader.pdf', figs_dir: str = '') -> None:
    settings = SimSettings(
        leader_state=leader,
        cars_params=tuple(
        (
            CarParameters(pos_i=pos),
            ControllerParameters(
                th=0.0,
            )
        ) for pos in [-13.5, -27.0, -40.5, -54.0]
        )
    )
    sim_data = Simulator(settings).sim_and_get_data()
    
    if not do_plots: return

    plot_car_positions(sim_data, include_leader=True)
    plt.savefig(os.path.join(figs_dir, fig_file))

    plt.show(block=False)

def ex_avg_speed_leader_constant_vel_to_next_car(do_plots: bool = True, figs_dir: str = '') -> None:
    ex_constant_dist_to_next_car(
        avg_speed_leader, 
        do_plots=do_plots, 
        figs_dir=figs_dir, 
        fig_file='ex_avg_speed_leader_constant_vel_to_next_car.pdf'
    )

def ex_slow_leader_constant_vel_to_next_car(do_plots: bool = True, figs_dir: str = '') -> None:
    ex_constant_dist_to_next_car(
        slow_leader, 
        do_plots=do_plots, 
        figs_dir=figs_dir, 
        fig_file='ex_slow_leader_constant_vel_to_next_car.pdf'
    )

def ex_fast_leader_constant_vel_to_next_car(do_plots: bool = True, figs_dir: str = '') -> None:
    ex_constant_dist_to_next_car(
        fast_leader, 
        do_plots=do_plots, 
        figs_dir=figs_dir, 
        fig_file='ex_fast_leader_constant_vel_to_next_car.pdf'
    )

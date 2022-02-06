from __future__ import annotations

import matplotlib.pyplot as plt

from simulation.sim_dataflow import SimData


def plot_car_positions(data: SimData, include_leader: bool = False, 
                        plot_references: bool = True, plot_control_signals: bool = True) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True)
    ax1, ax2 = axes
    # fill between car back and front
    for i in range(data.n_cars):
        color = "C" + str(i)
        ax1.fill_between(
            data.time, 
            data.positions[:, i],
            data.positions[:, i] + data.settings.cars_params[i][0].length, 
            alpha=0.8,
            color=color
        )
        if plot_references:
            ax1.plot(
                data.time,
                data.references[:, i],
                color=color,
                linestyle="--",
            )
        if plot_control_signals:
            ax2.plot(
                data.time,
                data.control_signals[:, i]/1000.0,
                color=color,
            )
    

    if include_leader:
        ax1.fill_between(
            data.time, 
            [data.settings.leader_state(t)[0] for t in data.time],
            [data.settings.leader_state(t)[0] + data.settings.cars_params[0][0].length for t in data.time], 
            alpha=0.8, 
            color="black"
        )
    
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("control signal (kNm)")
    ax1.set_ylabel("position (m)")

    